"""
Implementation of the LMR regularizer with the goal of learning a single image mask
Uses a similar process as the method outlined in the file LNR.py, but instead focuss on only a single image
"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.linalg import inv_ex
from torch.utils.data import Dataset


class Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.relu_layer = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu_layer(out)
        out = self.conv2(out)
        out = self.relu_layer(out)
        return out


class MaskLearner(nn.Module):
    """
    Class which actually does the masking
    This model treats the masking problem like a two-class segmentation classification problem
    - pixels to keep are one class, pixels to segment are another
    """

    def __init__(self, device: torch.device, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pool_layer = nn.MaxPool2d(2, 2)

        # Down Convolution blocks
        self.d_conv1 = Conv_Block(3, 8)
        self.d_conv2 = Conv_Block(8, 16)
        self.d_conv3 = Conv_Block(16, 32)
        self.d_conv4 = Conv_Block(32, 64)

        # Bottlneck
        self.b_neck = Conv_Block(64, 128)

        # Up Convolution blocks
        self.up_conv1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.u_conv1 = Conv_Block(128, 64)
        self.up_conv2 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.u_conv2 = Conv_Block(64, 32)
        self.up_conv3 = nn.ConvTranspose2d(32, 16, 2, 2)
        self.u_conv3 = Conv_Block(32, 16)
        self.up_conv4 = nn.ConvTranspose2d(16, 8, 2, 2)
        self.u_conv4 = Conv_Block(16, 8)

        # Final 1x1 convolution
        self.out_layer = nn.Conv2d(
            8, 2, 1
        )  # 2 pseudo-classes - index 0 is probability to remove and index 1 is the probability to keep
        self.to(device)

    def forward(self, x: torch.Tensor):
        """
        process a batch of samples through the mask learning model
        """
        skip_block_1 = self.d_conv1(x)
        out = self.pool_layer(skip_block_1)

        skip_block_2 = self.d_conv2(out)
        out = self.pool_layer(skip_block_2)

        skip_block_3 = self.d_conv3(out)
        out = self.pool_layer(skip_block_3)

        skip_block_4 = self.d_conv4(out)
        out = self.pool_layer(skip_block_4)

        out = self.b_neck(out)

        out = self.up_conv1(out)
        out = self.u_conv1(torch.cat((skip_block_4, out), dim=1))

        out = self.up_conv2(out)
        out = self.u_conv2(torch.cat((skip_block_3, out), dim=1))

        out = self.up_conv3(out)
        out = self.u_conv3(torch.cat((skip_block_2, out), dim=1))

        out = self.up_conv4(out)
        out = self.u_conv4(torch.cat((skip_block_1, out), dim=1))

        out = self.out_layer(out)

        return out

    @staticmethod
    def get_mask_from_logits(logits: torch.Tensor) -> torch.Tensor:
        """
        gets a mask of the top-k highest liklihood points to mask
        """
        # find top k probabilities in the mask field and mask them
        masked_prob_hat = logits[:, 0, ...]
        flat_probs = masked_prob_hat.view(masked_prob_hat.shape[0], -1)
        final_mask = torch.ones_like(masked_prob_hat)

        # select k highest probability pixels
        _, pixels_to_mask = torch.topk(flat_probs, k=10000)

        # reshape
        indices_for_pixels = torch.unravel_index(pixels_to_mask, masked_prob_hat.shape)

        # generate final mask
        final_mask[indices_for_pixels] = 0

        # max probability in index 0 means pixel should be masked,
        # this means the tensor can be returned directly
        return final_mask.long()

    @staticmethod
    def apply_mask_as_mixer(
        logits: torch.Tensor,
        batch_inputs: torch.Tensor,
        batch_targets: torch.Tensor,
        dataset: Dataset,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        given a mask and an input batch of images apply the mask to fill in data from other images
        """
        # select a random image in the dataset
        swap = random.choice(dataset)
        swap_image = torch.Tensor(swap["image"]).float().to("mps")
        swap_target = torch.Tensor(swap["depth"]).float().to("mps")
        swap_image = swap_image.permute(2, 0, 1).unsqueeze(
            0
        )  # put channel in first place

        # get mask from the unet logits
        mask = MaskLearner.get_mask_from_logits(logits)

        # select pixels from the mask from a different imagea
        inv_mask = torch.logical_not(
            mask.bool()
        )  # mask has 1 for kept and 0 for masked, we want the opposite to index

        # generate input image by combining
        # permute for broadcasting to work
        new_imgs = (
            batch_inputs.permute(1, 0, 2, 3) * inv_mask.float()
            + swap_image.permute(1, 0, 2, 3) * mask
        )
        new_targets = batch_targets * inv_mask.float() + swap_target * mask

        return new_imgs.permute(1, 0, 2, 3), new_targets

    @staticmethod
    def apply_mask_as_cutout(
        logits: torch.Tensor,
        batch_inputs: torch.Tensor,
        batch_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        cut out the mask from the provided images - this assumes that all colors will be assigned to black and
        the depths will be assigned to zero
            args:
                mask: torch.Tensor - tensor of image masks
                batch: torch.Tensor - tensor of input images
        """
        # here I will have logits with two dimensions
        # I need to select the pseudo-class which has the highest probability
        mask = MaskLearner.get_mask_from_logits(logits)

        # reshape batch for masking
        reshaped_batch = batch_inputs.permute(1, 0, 2, 3)
        reshaped_batch = (
            reshaped_batch * mask
        )  # multiply instead of indexing to keep gradient flow
        batch_targets = batch_targets * mask

        return batch_inputs.permute(0, 1, 2, 3), batch_targets

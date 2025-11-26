"""
Implementation of the LMR regularizer with the goal of learning a single image mask
Uses a similar process as the method outlined in the file LNR.py, but instead focuss on only a single image
"""

import torch
import torch.nn as nn


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
        self.out_layer = nn.Conv2d(8, 2, 1)  # 2 pseudo-classes for mask and don't mask
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

    def apply_mask_as_mixer(self) -> None:
        """
        given a mask and an input batch of images apply the mask to fill in data from other images
        """
        pass

    def apply_mask_as_cutout(
        self,
        mask: torch.Tensor,
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
        mask = mask.argmax(dim=1)

        # reshape batch for masking
        reshaped_batch = batch_inputs.permute(1, 0, 2, 3)
        reshaped_batch = (
            reshaped_batch * mask
        )  # multiply instead of indexing to keep gradient flow
        batch_targets = batch_targets * 0.0

        return batch_inputs.permute(0, 1, 2, 3), batch_targets

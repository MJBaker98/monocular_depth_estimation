"""
This file contains a library of loss methods which can be used to optimize the learned mask regularizer (LMR)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._refs import zero


class LMRLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(
        self,
        net_mask: torch.Tensor,
        depth_hat: torch.Tensor,
        depth: torch.Tensor,
        k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Implementing the forward method as defined in the paper which relies on information gain
        """
        return self.info_gain_loss(net_mask, depth_hat, depth, k)

    def as_logits(self, logits: torch.Tensor, class_labels: torch.Tensor):
        """
        Apply cross entropy loss treating the mask alignment as a classification problem
        """
        mask_weight = 480.0 * 640 / 10000  # size of image/num of pixels to mask
        weight_tensor = torch.Tensor([mask_weight, 1.0]).to("mps")

        # weight is applied to 0 becuse the zero indicates the pixel is removed
        # i.e., there are going to be way more kept pixels than removed ones,
        # so weight the removed pixels more
        return F.cross_entropy(weight=weight_tensor, input=logits, target=class_labels)

    def as_probabilities(self, logits: torch.Tensor, class_labels: torch.Tensor):
        """
        Use Binary Cross Entropy using a sigmoid activation on the logits
        """
        return F.binary_cross_entropy_with_logits(logits, target=class_labels)

    def info_gain_loss(
        self,
        net_mask: torch.Tensor,
        depth_hat: torch.Tensor,
        depth: torch.Tensor,
        k: int = 10000,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Original idea usees information gain like that is used to build decision trees"""
        # calculate d_hat - d
        diff = torch.abs(depth_hat - depth)

        # ignore depth values where the truth is zero (i.e., set them to some large value so probability is low)
        zero_depth_mask = (depth == 0).bool()
        diff[zero_depth_mask] = 1000

        # predict liklihood of next iteration via gaussian activation function
        gauss_pred = self.gaussian_activation(diff)

        # find top-k pixels
        assert k > 0
        gauss_pred_flat = gauss_pred.reshape(gauss_pred.shape[0], -1)
        _, top_k_inds_flat = torch.topk(gauss_pred_flat, k=k, dim=1)

        # create two matrices representing the image-by-image masks
        pred_mask_keep = torch.ones_like(gauss_pred)
        top_k_inds = torch.unravel_index(top_k_inds_flat, gauss_pred.shape)

        # we want the pred_mask_keep to be 0 for all pixels in the top_k_inds and 1
        # for the other pixels. This is a pseudo ('truth') psuedo class setup
        pred_mask_keep[top_k_inds] = 0.0  # 'mask out' top k value in keep mask
        pred_mask_keep = pred_mask_keep.long()

        # # inverse of the keep tensor
        # pred_mask_discard = torch.abs(pred_mask_keep - 1)
        # pred_mask_discard = pred_mask_discard.long()

        # # stack probabilities
        # pred_probs = torch.cat(
        #     [pred_mask_discard.unsqueeze(0), pred_mask_keep.unsqueeze(1)], dim=1
        # )

        # compare results using binary cross entropy and cross entropy
        cel = self.as_logits(net_mask, pred_mask_keep)
        # bcel = self.as_probabilities(net_mask, pred_probs.float())
        return cel, pred_mask_keep

    def gaussian_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Calcualte gaussian activation function e^(-x^2)"""
        return torch.exp(-1 * x**2)

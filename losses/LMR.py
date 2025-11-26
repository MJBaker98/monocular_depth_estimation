"""
This file contains a library of loss methods which can be used to optimize the learned mask regularizer (LMR)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LMRLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(
        self,
        net_mask: torch.Tensor,
        depth_hat: torch.Tensor,
        depth: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """
        Implementing the forward method as defined in the paper which relies on information gain
        """
        return self.info_gain_loss(net_mask, depth_hat, depth, k)

    def info_gain_loss(
        self,
        net_mask: torch.Tensor,
        depth_hat: torch.Tensor,
        depth: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """Original idea usees information gain like that is used to build decision trees"""

        # get mask from pseudo-classes
        # here a 1 means the pixel is kept, 0 means it is masked
        # I think we may need to reverse this to get training to work
        mask = net_mask.softmax(dim=1)

        # calculate d_hat - d
        diff = torch.abs(depth_hat - depth)

        # predict liklihood of next iteration via gaussian activation function
        gauss_pred = self.gaussian_activation(diff)

        # find top-k pixels
        assert k > 0
        gauss_pred_flat = gauss_pred.reshape(gauss_pred.shape[0], -1)
        _, top_k_inds_flat = torch.topk(gauss_pred_flat, k=k, dim=1)

        # create two matrices representing the image-by-image masks
        pred_mask_keep = torch.ones_like(gauss_pred)
        top_k_inds = torch.unravel_index(top_k_inds_flat, gauss_pred.shape)

        # we want the pred_mask_keep to be 1 for all pixels in the top_k_inds and zero
        # for the other pixels. We want all indices in pred_mask_mask to be 0 except for
        # the pixels to mask which would be 1. This is a pseudo ('truth') psuedo class setup
        pred_mask_keep[top_k_inds] = 0.0  # 'mask out' top k value in keep mask
        pred_mask_keep = pred_mask_keep.long()

        # pred_mask_mask = torch.abs(pred_mask_keep - 1).long()  # get the reverse set of points

        # # unsqueeze each mask to get it in batch, channel, [shape] format
        # pred_mask_keep = pred_mask_keep.unsqueeze(1)
        # pred_mask_mask = pred_mask_mask.unsqueeze(1)
        # pseudo_probability = torch.cat(
        #     (pred_mask_keep, pred_mask_mask), dim=1
        # )  # cat along probability dimension

        # iou = torch.sum(mask * pseudo_probability) / torch.sum(
        #     (mask * pred_mask_keep)
        #     + (mask * pred_mask_mask)
        #     - (mask * pseudo_probability)
        # )
        # print(f"iou: {iou.item()}")
        cel = F.cross_entropy(net_mask, pred_mask_keep)
        # iou_loss = torch.log(1 / (iou + 1e-6))
        return cel

    def gaussian_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Calcualte gaussian activation function e^(-x^2)"""
        return torch.exp(-1 * x**2)

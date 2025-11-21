"""
This file contains a library of loss methods which can be used to optimize the learned mask regularizer (LMR)

"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import override


class LMRLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @override
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

        # calculate d_hat - d
        diff = torch.abs(depth_hat - depth)

        # predict liklihood of next iteration via gaussian activation function
        gauss_pred = self.gaussian_activation(diff)

        # find top-k pixels
        assert k > 0
        _, top_k_inds = torch.topk(gauss_pred, k)

        net_mask = net_mask.to(torch.bool)
        pred_mask = top_k_inds.to(torch.bool)

        iou = torch.sum(net_mask * pred_mask) / torch.sum(net_mask + pred_mask)
        return torch.log(1 / iou)

    def gaussian_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Calcualte gaussian activation function e^(-x^2)"""
        return torch.exp(-1 * x ^ 2)

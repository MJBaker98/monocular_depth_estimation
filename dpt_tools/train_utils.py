from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL.TiffImagePlugin import ImageFileDirectory
from torch.utils.data import Dataset

from models.LMR import MaskLearner


def regression_cutmix(
    images: torch.Tensor, targets: torch.Tensor, beta: float = 0.1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Method that implements the CutMix regularizer for regression tasks
        Much of this follows the implementation from the CutMix-PyTorch repo

    Args:
        images: torch.Tensor, Images to process
        targets: torch.Tensor, real value points to transfer (e.g., depth)

    Returns:
        transformed_images: torch.Tensor, Images with replaced regions
        transformed_targets: torch.Tensor, value maps with replaced regions
    """
    lam = np.random.beta(beta, beta)
    rand_index = torch.randperm(images.size()[0]).to("mps")
    target_a = targets
    target_b = targets[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
    images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]

    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))

    return lam, images, target_a, target_b


def rand_bbox(size, lam):
    """
    random bounding box code from CutMix Pytorch library
    """
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def plot_test_frames(
    model: nn.Module,
    dataset: Dataset,
    indices: List[int],
    epoch: int,
    save_fig: bool = False,
) -> None:
    """Generate a plot of depth images at specific indices"""
    for i in indices:
        datapoint = dataset[i]
        X = datapoint["image"]
        y = datapoint["depth"]
        mask = datapoint["mask"]

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 8))
        ax1.imshow(X)
        ax1.set_title("Image")
        ax2.imshow(y)
        ax2.set_title("Truth depth")

        X = torch.Tensor(X).to("mps").unsqueeze(0).permute(0, 3, 1, 2)
        with torch.no_grad():
            prediction = model(X).permute(1, 2, 0).cpu().numpy()
        ax3.imshow(prediction, cmap="viridis")
        ax3.set_title("Predicted depth")

        output_path = Path("output/figs")

        if save_fig:
            out_str = f"depth_index_{i}_epoch_{epoch}.png"
            plt.savefig(out_str)
        else:
            plt.show()


def plot_while_training(
    image: torch.Tensor,
    truth: torch.Tensor,
    prediction: torch.Tensor,
    epoch: int,
    model_name: str,
) -> None:
    fig, ((ax1, ax2, _), (ax3, ax4, ax5)) = plt.subplots(2, 3, figsize=(10, 8))
    ax1.imshow(image.permute(1, 2, 0).cpu())
    ax1.set_title("Image")
    im_ax2 = ax2.imshow(truth.cpu(), cmap="viridis")
    ax2.set_title("Truth depth")
    plt.colorbar(im_ax2)

    im_ax3 = ax3.imshow(prediction.cpu(), cmap="viridis")
    ax3.set_title("Predicted depth")
    plt.colorbar(im_ax3)

    diff = torch.abs(truth - prediction).cpu()

    im_ax4 = ax4.imshow(diff, cmap="viridis")
    ax4.set_title("Depth difference")
    plt.colorbar(im_ax4)

    im_ax5 = ax5.imshow(torch.exp(-1 * diff**2), cmap="viridis")
    ax5.set_title("Probability from depth difference")
    plt.colorbar(im_ax5)

    out_path = Path(f"output/figs/{model_name}")
    out_path.mkdir(parents=True, exist_ok=True)
    out_path = out_path / f"depth_index_epoch_{epoch}.png"
    plt.savefig(out_path)

    # clean up figures
    plt.close()
    del fig


def plot_lmr_mask(
    image: torch.Tensor,
    net_logits: torch.Tensor,
    depth_based_mask: torch.Tensor,
    epoch: int,
):
    """
    Plot function specifically for the mask
    """
    mask_mask = MaskLearner.get_mask_from_logits(net_logits, 10000)
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(10, 8))

    # use first image
    image_id = 0

    img_to_show = image[image_id, ...]  # select image_id from batch
    mask_to_show = mask_mask[image_id, ...]
    depth_mask_to_show = depth_based_mask[image_id, ...]

    ax0.imshow(img_to_show.permute(1, 2, 0).cpu())
    ax0.set_title("Image color")
    ax1.imshow(mask_to_show.cpu(), cmap="binary")
    ax1.set_title("Learned Mask")
    ax2.imshow(depth_mask_to_show.cpu(), cmap="binary")
    ax2.set_title("Accuracy-driven Mask")

    out_path = Path("output/figs/LMR/lmr_mask")
    out_path.mkdir(parents=True, exist_ok=True)
    out_path = out_path / f"lmr_mask_comparison_epoch_{epoch}.png"
    plt.savefig(out_path)
    plt.close()
    del fig

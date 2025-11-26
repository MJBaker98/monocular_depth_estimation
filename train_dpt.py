"""
This file contains the classes necessary to implement the DPT fusion model
first published in Ranftl et. al., (Vision Transformers for Dense Prediction)
"""

import random
import sys
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from dataloaders.nyu_data import NyuDepthV2
from DPT.dpt.models import DPTDepthModel
from LNRegularizer.LNR import LNR
from losses.LMR import LMRLoss
from losses.mde_losses import ScaleAndShiftInvariantLoss

if "IPython" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

from datetime import datetime
from pathlib import Path

from models.LMR import MaskLearner
from models.utils import regression_cutmix


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
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 8))
    ax1.imshow(image.permute(1, 2, 0).cpu())
    ax1.set_title("Image")
    ax2.imshow(truth.cpu(), cmap="viridis")
    ax2.set_title("Truth depth")
    ax3.imshow(prediction.cpu(), cmap="viridis")
    ax3.set_title("Predicted depth")

    out_path = Path(f"output/figs/{model_name}")
    out_path.mkdir(parents=True, exist_ok=True)
    out_path = out_path / f"depth_index_epoch_{epoch}.png"
    plt.savefig(out_path)

    # clean up figures
    plt.close()
    del fig


def train_simple(
    model: nn.Module,
    loader: DataLoader,
    optim: Optimizer,
    scheduler: CosineAnnealingLR,
    epochs: int = 50,
    save_every: int = 10,
) -> None:
    """
    Train depth head on NYU dataset with no additional regularization
    """
    model.train()
    pbar = tqdm(total=epochs * len(loader), desc="Training MDE:")
    i = 0
    # loss function
    loss = ScaleAndShiftInvariantLoss()

    # training loop
    for e in range(epochs):
        for batch in loader:
            X = batch["image"].float().to("mps")
            y = batch["depth"].float().to("mps")
            mask = batch["mask"].float().to("mps")

            X = X.permute(0, 3, 1, 2)

            # calculate depth
            prediction = model(X)

            # calculate losses
            err = loss(prediction, y, mask)
            mse_loss = F.mse_loss(prediction, y)
            l1_loss = F.smooth_l1_loss(prediction, y)

            composite_loss = (2 * err) + (0.5 * mse_loss) + (0.1 * l1_loss)

            # print(f"composite_loss requires_grad: {composite_loss.requires_grad}")

            # process optimizer
            optim.zero_grad()
            composite_loss.backward()  # back-prop losses
            optim.step()
            scheduler.step()

            # debugging
            grads = []
            for name, p in model.named_parameters():
                if p.grad is not None:
                    grads.append(p.grad.norm().item())
                else:
                    pass
                    # print(f"{name} gradient is none")

            grads = torch.Tensor(grads)

            if i % 1 == 0:
                with torch.no_grad():
                    mse_loss = F.mse_loss(prediction, y)
                pbar.set_postfix_str(
                    f"train_loss: {err:.2f} | mse_loss: {mse_loss:.2f} | l1_loss: {l1_loss:.2f} | composite loss: {composite_loss:.2f} | min pred. depth: {prediction.min().item():.2f} | max pred. depth: {prediction.max().item():.2f}"
                )
                pbar.update(1)

            i += 1

        with torch.no_grad():
            try:
                plot_while_training(
                    X[0, ...], y[0, ...], prediction[0, ...], e, "simple"
                )
            except:
                print("Failed while writing figure... continuing")

        if e % save_every == 0:
            print(f"Saving checkpoint at epoch {e}:")
            simple_path = f"output/checkpoint/simple_model_epoch_{e}.pth"
            torch.save(model.state_dict(), simple_path)


def train_with_lmr(
    model: nn.Module,
    mask_learning_model: nn.Module,
    loader: DataLoader,
    optim: Optimizer,
    scheduler: CosineAnnealingLR | None,
    epochs: int = 50,
    save_every: int = 10,
) -> None:
    """
    Train depth head on NYU dataset with lmr regularizer
    """
    model.train()
    pbar = tqdm(total=epochs * len(loader), desc="Training MDE:")
    i = 0
    # loss function
    loss = ScaleAndShiftInvariantLoss()
    lmr_loss = LMRLoss()

    # training loop
    for e in range(epochs):
        for batch in loader:
            X = batch["image"].float().to("mps")
            y = batch["depth"].float().to("mps")
            mask = batch["mask"].float().to("mps")

            X = X.permute(0, 3, 1, 2)

            # pass input through LMR model
            logits = mask_learning_model(X)

            # apply the mask to the image
            # X, y = LMR_model.apply_mask_as_cutout(
            #     mask=logits, batch_inputs=X, batch_targets=y
            # )

            # calculate depth
            prediction = model(X)

            # calculate losses
            err = loss(prediction, y, mask)
            mse_loss = F.mse_loss(prediction, y)
            l1_loss = F.smooth_l1_loss(prediction, y)
            lmr_mask_loss = 1000 * lmr_loss(
                net_mask=logits, depth_hat=prediction, depth=y, k=10000
            )

            composite_loss = (0.2 * err) + (lmr_mask_loss)  # combine losses

            # process optimizer
            optim.zero_grad()
            composite_loss.backward()  # back-prop losses
            optim.step()
            if scheduler:
                scheduler.step()

            if i % 1 == 0:
                with torch.no_grad():
                    mse_loss = F.mse_loss(prediction, y)
                pbar.set_postfix_str(
                    f"train_loss: {err:.2f} | mse_loss: {mse_loss:.2f} | l1_loss: {l1_loss:.2f} | LMR Loss: {lmr_mask_loss:.2f} | composite loss: {composite_loss:.2f} | min pred. depth: {prediction.min().item():.2f} | max pred. depth: {prediction.max().item():.2f}"
                )
                pbar.update(1)

            i += 1

        with torch.no_grad():
            try:
                plot_while_training(X[0, ...], y[0, ...], prediction[0, ...], e, "LMR")
            except:
                print("Failed while writing figure... continuing")

        if e % save_every == 0:
            print(f"Saving checkpoint at epoch {e}:")
            simple_path = f"output/checkpoint/cutmix_model_epoch_{e}.pth"
            torch.save(model.state_dict(), simple_path)


def train_with_cutmix(
    model: nn.Module,
    loader: DataLoader,
    optim: Optimizer,
    scheduler: CosineAnnealingLR,
    epochs: int = 50,
    save_every: int = 10,
    cutmix_probability: float = 0.1,
) -> None:
    """
    Train depth head on NYU dataset using cutmix regularization
    """
    model.train()
    pbar = tqdm(total=epochs * len(loader), desc="Training MDE:")
    i = 0
    # loss function
    loss = ScaleAndShiftInvariantLoss()

    # training loop
    for e in range(epochs):
        for batch in loader:
            images = batch["image"].float().to("mps")
            targets = batch["depth"].float().to("mps")
            mask = batch["mask"].float().to("mps")

            images = images.permute(0, 3, 1, 2)

            if random.random() <= cutmix_probability:
                # apply cutmix to batch with a specifiic probability
                lam, images, y_a, y_b = regression_cutmix(images, targets)

                # calculate depth
                prediction = model(images)

                # call the loss function on each output and input set separately
                def apply_cutmix(fcn: Callable) -> torch.Tensor:
                    return fcn(prediction, y_a) * lam + fcn(prediction, y_b) * (
                        1.0 - lam
                    )

                # calculate losses
                err = loss(prediction, y_a, mask) * lam + loss(
                    prediction, y_b, mask
                ) * (1.0 - lam)
                mse_loss = apply_cutmix(F.mse_loss)
                l1_loss = apply_cutmix(F.smooth_l1_loss)
            else:
                prediction = model(images)
                err = loss(prediction, targets, mask)
                mse_loss = F.mse_loss(prediction, targets)
                l1_loss = F.smooth_l1_loss(prediction, targets)

            composite_loss = (0.1 * err) + (0.5 * mse_loss) + (0.1 * l1_loss)

            # process optimizer
            optim.zero_grad()
            composite_loss.backward()  # back-prop losses
            optim.step()
            scheduler.step()

            if i % 1 == 0:
                with torch.no_grad():
                    mse_loss = F.mse_loss(prediction, targets)
                pbar.set_postfix_str(
                    f"train_loss: {err:.2f} | mse_loss: {mse_loss:.2f} | l1_loss: {l1_loss:.2f} | composite loss: {composite_loss:.2f} | min pred. depth: {prediction.min().item():.2f} | max pred. depth: {prediction.max().item():.2f}"
                )
                pbar.update(1)

            i += 1

        with torch.no_grad():
            try:
                plot_while_training(
                    images[0, ...], targets[0, ...], prediction[0, ...], e, "cutmix"
                )
            except:
                print("Failed while writing figure... continuing")

        if e % save_every == 0:
            print(f"Saving checkpoint at epoch {e}:")
            simple_path = f"output/checkpoint/cutmix_model_epoch_{e}.pth"
            torch.save(model.state_dict(), simple_path)


def eval(model: nn.Module, loader: DataLoader) -> Dict:
    """
    assess the accuracy of the model
    """
    model.eval()
    pbar = tqdm(total=len(loader), desc="Evaluating MDE:")
    test_err = []
    for batch in loader:
        X = batch["image"].float().to("mps")
        y = batch["depth"].float().to("mps")
        with torch.no_grad():
            X = X.permute(0, 3, 1, 2)
            prediction = model(X)
            err = F.mse_loss(
                prediction, y
            )  # when looking at error for evaluation use MSE loss
            test_err.append(err)

        # process optimizer
        pbar.set_postfix_str(f"mse_loss: {err:.2f}")
        pbar.update(1)

    print(f"Average MSE Loss: {sum(test_err) / len(test_err)}")
    return {"mse_avg": sum(test_err) / len(test_err)}


def init_model():
    model = (
        DPTDepthModel(
            path="/Users/michael/Documents/Grad_School/Fall25/DeepLearning/Project/MDE/DPT/dpt/weights/dpt_hybrid-midas-501f0c75.pt",
            scale=0.000305,
            shift=0.1378,
            invert=True,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        .float()
        .to("mps")
    )

    # keep everything but initialize the final head model
    for name, param in model.named_parameters():
        if "output_conv" in name or "head" in name:
            if "weight" in name:
                print(f"Initializing {name} to kaiming normal")
                _ = nn.init.kaiming_normal_(param)

    return model


if __name__ == "__main__":
    print("Running training and assessment for DPT-based model")

    NYU_DATA_PATH = "data/nyu_data/nyu_depth_v2_labeled.mat"

    # download from http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat
    NYU_SPLIT_PATH = "data/nyu_data/splits.mat"

    nyu_test_ds = NyuDepthV2(NYU_DATA_PATH, NYU_SPLIT_PATH, split="test")
    nyu_train_ds = NyuDepthV2(NYU_DATA_PATH, NYU_SPLIT_PATH, split="train")
    nyu_train_dataloader = DataLoader(nyu_train_ds, batch_size=12)
    nyu_test_dataloader = DataLoader(nyu_test_ds, batch_size=12)

    ########################
    # model training booleans
    do_simple = False
    do_cutmix = False
    do_LMR = True

    ########################
    # Model agnostic hyperparameters
    epochs = 5

    ########################
    # Simple model
    if do_simple:
        simple_model = init_model()

        optim = Adam(simple_model.parameters(), lr=1e-5)
        scheduler = CosineAnnealingLR(optim, eta_min=1e-8, T_max=epochs)

        # standard training - no regularization at all
        train_simple(
            model=simple_model,
            loader=nyu_train_dataloader,
            optim=optim,
            epochs=epochs,
            scheduler=scheduler,
        )
        simple_res = eval(simple_model, nyu_test_dataloader)
        timestr = datetime.now().strftime("%a_%d_%b_%Y_%I:%M%p")
        simple_path = "output/checkpoint/simple_model_" + timestr + ".pth"
        torch.save(simple_model.state_dict(), simple_path)

    #######################
    # Cutmix model
    if do_cutmix:
        cutmix_model = init_model()

        optim = Adam(cutmix_model.parameters(), lr=1e-4)
        scheduler = CosineAnnealingLR(optim, eta_min=1e-8, T_max=epochs)

        # standard training - no regularization at all
        train_with_cutmix(
            model=cutmix_model,
            loader=nyu_train_dataloader,
            optim=optim,
            epochs=5,
            cutmix_probability=0.25,
            scheduler=scheduler,
        )
        cutmix_res = eval(cutmix_model, nyu_test_dataloader)
        timestr = datetime.now().strftime("%a_%d_%b_%Y_%I:%M%p")
        cutmix_path = "output/checkpoint/cutmix_model" + timestr + ".pth"
        torch.save(cutmix_model.state_dict(), cutmix_path)

    #######################
    # LMR model
    if do_LMR:
        lmr_model = init_model()

        optim = Adam(lmr_model.parameters(), lr=1e-5)
        scheduler = CosineAnnealingLR(optim, eta_min=1e-8, T_max=epochs)

        # standard training - no regularization at all
        train_with_lmr(
            model=lmr_model,
            loader=nyu_train_dataloader,
            optim=optim,
            epochs=5,
            scheduler=scheduler,
        )
        lmr_res = eval(lmr_model, nyu_test_dataloader)
        timestr = datetime.now().strftime("%a_%d_%b_%Y_%I:%M%p")
        lmr_path = "output/checkpoint/lmr_model" + timestr + ".pth"
        torch.save(lmr_model.state_dict(), lmr_path)

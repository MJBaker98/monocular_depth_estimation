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
from models.LMR import MaskLearner

if "IPython" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

from datetime import datetime
from pathlib import Path

from dpt_tools.train_utils import (
    plot_lmr_mask,
    plot_test_frames,
    plot_while_training,
    regression_cutmix,
)


def train_simple(
    model: nn.Module,
    loader: DataLoader,
    optim: Optimizer,
    scheduler: CosineAnnealingLR,
    epochs: int = 50,
    save_every: int = 10,
) -> str:
    """
    Train depth head on NYU dataset with no additional regularization
    """
    model.train()
    pbar = tqdm(total=epochs * len(loader), desc="Training MDE:")
    i = 0
    # loss function
    loss = ScaleAndShiftInvariantLoss()
    timestamp = datetime.now().strftime("%a_%d_%b_%Y_%I_%M%p")

    log_path = "output/log"
    logfile_name = log_path + "/Log_Simple_" + timestamp + ".txt"

    # training loop
    for e in range(epochs):
        errs = []
        mse_losses = []
        l1_losses = []
        composite_losses = []
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

            # Record losses
            errs.append(err)
            mse_losses.append(mse_loss)
            l1_losses.append(l1_loss)
            composite_losses.append(composite_loss)

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
            output_path = Path(log_path)
            output_path.mkdir(parents=True, exist_ok=True)
            with open(logfile_name, "a") as logfile:
                logfile.write(
                    f"epoch: {e} | train_loss: {sum(errs) / len(errs):.2f} | mse_loss: {sum(mse_losses) / len(mse_losses):.2f} | l1_loss: {sum(l1_losses) / len(l1_losses):.2f} | composite loss: {sum(composite_losses) / len(composite_losses):.2f}\n"
                )
            try:
                plot_test_frames(
                    model,
                    dataset=loader.dataset,
                    indices=[1, 3, 5],
                    epoch=e,
                    model_name="simple" + timestamp,
                    save_fig=True,
                )
                plot_while_training(
                    X[0, ...], y[0, ...], prediction[0, ...], e, "simple" + timestamp
                )
            except:
                print("Failed while writing figure... continuing")

        if e % save_every == 0:
            print(f"Saving checkpoint at epoch {e}:")
            simple_path = f"output/checkpoint/simple_model_epoch_{e}.pth"
            torch.save(model.state_dict(), simple_path)
    return logfile_name


def train_with_lmr(
    model: nn.Module,
    mask_learning_model: nn.Module,
    loader: DataLoader,
    optim: Optimizer,
    scheduler: CosineAnnealingLR | None,
    epochs: int = 50,
    save_every: int = 10,
    visualize_mask: bool = False,
) -> str:
    """
    Train depth head on NYU dataset with lmr regularizer
    """
    model.train()
    pbar = tqdm(total=epochs * len(loader), desc="Training MDE:")
    i = 0
    # loss function
    loss = ScaleAndShiftInvariantLoss()
    lmr_loss = LMRLoss()

    timestamp = datetime.now().strftime("%a_%d_%b_%Y_%I_%M%p")
    log_path = "output/log"
    logfile_name = log_path + "/Log_LMR_" + timestamp

    # training loop
    for e in range(epochs):
        errs = []
        mse_losses = []
        l1_losses = []
        lmr_mask_losses = []
        composite_losses = []
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
            lmr_mask_loss, depth_difference_mask = lmr_loss(
                net_mask=logits, depth_hat=prediction.detach(), depth=y, k=10000
            )
            mse_loss = F.mse_loss(prediction, y)
            l1_loss = F.smooth_l1_loss(prediction, y)

            composite_loss = (1.0 * err) + (1.1 * lmr_mask_loss)  # combine losses

            # Record losses
            errs.append(err)
            mse_losses.append(mse_loss)
            l1_losses.append(l1_loss)
            lmr_mask_losses.append(lmr_mask_loss)
            composite_losses.append(composite_loss)

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
                    f"train_loss: {err:.2f} | mse_loss: {mse_loss:.2f} | l1_loss: {l1_loss:.2f} | LMR Loss: {lmr_mask_loss:.3f} | composite loss: {composite_loss:.2f} | min pred. depth: {prediction.min().item():.2f} | max pred. depth: {prediction.max().item():.2f}"
                )
                pbar.update(1)

            i += 1

        with torch.no_grad():
            output_path = Path(log_path)
            output_path.mkdir(parents=True, exist_ok=True)
            with open(logfile_name, "a") as logfile:
                logfile.write(
                    f"epoch: {e} | train_loss: {sum(errs) / len(errs):.2f} | mse_loss: {sum(mse_losses) / len(mse_losses):.2f} | l1_loss: {sum(l1_losses) / len(l1_losses):.2f} | LMR Loss: {sum(lmr_mask_losses) / len(lmr_mask_losses):.2f} | composite loss: {sum(composite_losses) / len(composite_losses):.2f}\n"
                )
            try:
                plot_while_training(X[0, ...], y[0, ...], prediction[0, ...], e, "LMR")
                # plot_test_frames(
                #     model, loader.dataset, [1, 3, 5], e, model_name="LMR_" + timestamp
                # )
                if visualize_mask:
                    plot_lmr_mask(
                        image=X,
                        net_logits=logits,
                        depth_based_mask=depth_difference_mask,
                        epoch=e,
                    )
            except Exception as e:
                print(f"Failed while writing figure with error {e} ... continuing")

        if e % save_every == 0:
            print(f"Saving checkpoint at epoch {e}:")
            lmr_cpt_path = f"output/checkpoint/lmr_model_epoch_{e}.pth"
            torch.save(model.state_dict(), lmr_cpt_path)
    return logfile_name


def train_with_cutmix(
    model: nn.Module,
    loader: DataLoader,
    optim: Optimizer,
    scheduler: CosineAnnealingLR,
    epochs: int = 50,
    save_every: int = 10,
    cutmix_probability: float = 0.1,
) -> str:
    """
    Train depth head on NYU dataset using cutmix regularization
    """
    model.train()
    pbar = tqdm(total=epochs * len(loader), desc="Training MDE:")
    i = 0
    # loss function
    loss = ScaleAndShiftInvariantLoss()

    timestamp = datetime.now().strftime("%a_%d_%b_%Y_%I_%M%p")
    log_path = "output/log"
    logfile_name = log_path + "/Log_Cutmix_" + timestamp

    # training loop
    for e in range(epochs):
        errs = []
        mse_losses = []
        l1_losses = []
        composite_losses = []
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

            # composite_loss = (0.1 * err) + (0.5 * mse_loss) + (0.1 * l1_loss)
            composite_loss = err  # only use the shift and scale invariant loss

            # Record losses
            errs.append(err)
            mse_losses.append(mse_loss)
            l1_losses.append(l1_loss)
            composite_losses.append(composite_loss)

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
            output_path = Path(log_path)
            output_path.mkdir(parents=True, exist_ok=True)
            with open(logfile_name, "a") as logfile:
                logfile.write(
                    f"epoch: {e} | train_loss: {sum(errs) / len(errs):.2f} | mse_loss: {sum(mse_losses) / len(mse_losses):.2f} | l1_loss: {sum(l1_losses) / len(l1_losses):.2f} | composite loss: {sum(composite_losses) / len(composite_losses):.2f}\n"
                )
            try:
                plot_test_frames(
                    model, loader.dataset, [1, 3, 5], e, "cutmix" + timestamp, True
                )
                plot_while_training(
                    images[0, ...],
                    targets[0, ...],
                    prediction[0, ...],
                    e,
                    "cutmix" + timestamp,
                )
            except:
                print("Failed while writing figure... continuing")

        if e % save_every == 0:
            print(f"Saving checkpoint at epoch {e}:")
            simple_path = f"output/checkpoint/cutmix_model_epoch_{e}.pth"
            torch.save(model.state_dict(), simple_path)
    return logfile_name


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
        # create dpt model from DPT codebase
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

    MDE_model = DPTDepthModel(
        scale=0.000305,
        shift=0.1378,
        invert=True,
        backbone="vitb_rn50_384",
        non_negative=True,
        enable_attention_hooks=False,
    )  # create a standard DPT depth prediction model
    MDE_model = MDE_model.float().to("mps")
    NYU_DATA_PATH = "data/nyu_data/nyu_depth_v2_labeled.mat"

    # download from http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat
    NYU_SPLIT_PATH = "data/nyu_data/splits.mat"

    nyu_test_ds = NyuDepthV2(NYU_DATA_PATH, NYU_SPLIT_PATH, split="test")
    nyu_train_ds = NyuDepthV2(NYU_DATA_PATH, NYU_SPLIT_PATH, split="train")
    nyu_train_dataloader = DataLoader(nyu_train_ds, batch_size=4)
    nyu_test_dataloader = DataLoader(nyu_train_ds, batch_size=4)

    # Make output folder
    output_path = Path("output/checkpoint")
    output_path.mkdir(parents=True, exist_ok=True)

    ########################
    # model training runs
    do_simple = 5
    do_cutmix = 0
    do_LMR = 0

    ########################
    # Model agnostic hyperparameters
    epochs = 20

    ########################
    # Simple model
    for i in range(do_simple):
        print(f"Starting simple model {i}")
        simple_model = init_model()

        optim = Adam(simple_model.parameters(), lr=1e-4)
        scheduler = CosineAnnealingLR(optim, eta_min=1e-8, T_max=epochs)

        # Initial performance
        plot_test_frames(
            simple_model,
            nyu_train_dataloader.dataset,
            [1, 3, 5],
            i,
            "simple_before_training",
            True,
        )

        # standard training - no regularization at all
        log_name = train_simple(
            model=simple_model,
            loader=nyu_train_dataloader,
            optim=optim,
            epochs=epochs,
            scheduler=scheduler,
        )
        simple_res = eval(simple_model, nyu_test_dataloader)
        with open(log_name, "a") as file:
            file.write(f"Eval avg_mse: {simple_res['mse_avg']}")
        timestr = datetime.now().strftime("%a_%d_%b_%Y_%I_%M%p")
        simple_path = "output/checkpoint/simple_model_" + timestr + ".pth"
        torch.save(simple_model.state_dict(), simple_path)

    #######################
    # Cutmix model
    for i in range(do_cutmix):
        print(f"Starting Cutmix model {i}")
        cutmix_model = init_model()

        optim = Adam(cutmix_model.parameters(), lr=1e-4)
        scheduler = CosineAnnealingLR(optim, eta_min=1e-8, T_max=epochs)

        # Initial performance
        plot_test_frames(
            cutmix_model,
            nyu_train_dataloader.dataset,
            [1, 3, 5],
            i,
            "cutmix_before_training",
            True,
        )

        # Cutmix training
        log_name = train_with_cutmix(
            model=cutmix_model,
            loader=nyu_train_dataloader,
            optim=optim,
            epochs=epochs,
            cutmix_probability=0.25,
            scheduler=scheduler,
        )
        cutmix_res = eval(cutmix_model, nyu_test_dataloader)
        with open(log_name, "a") as file:
            file.write(f"Eval avg_mse: {cutmix_res['mse_avg']}")
        timestr = datetime.now().strftime("%a_%d_%b_%Y_%I_%M%p")
        cutmix_path = "output/checkpoint/cutmix_model" + timestr + ".pth"
        torch.save(cutmix_model.state_dict(), cutmix_path)

    #######################
    # LMR model
    for i in range(do_LMR):
        print(f"Starting LMR model {i}")
        lmr_model = init_model()

        optim = Adam(lmr_model.parameters(), lr=1e-4)
        scheduler = CosineAnnealingLR(optim, eta_min=1e-8, T_max=epochs)

        # Initial performance
        plot_test_frames(
            lmr_model,
            nyu_train_dataloader.dataset,
            [1, 3, 5],
            i,
            "lmr_before_training",
            True,
        )

        # Learned Mask Regularizer training
        log_name = train_with_lmr(
            model=lmr_model,
            mask_learning_model=LMR_model,
            loader=nyu_single_image_dataloader,
            optim=optim,
            epochs=epochs,
            scheduler=None,
            save_every=100,
            visualize_mask=True,
        )
        lmr_res = eval(lmr_model, nyu_test_dataloader)
        with open(log_name, "a") as file:
            file.write(f"Eval avg_mse: {lmr_res['mse_avg']}")
        timestr = datetime.now().strftime("%a_%d_%b_%Y_%I_%M%p")
        lmr_path = "output/checkpoint/lmr_model" + timestr + ".pth"
        torch.save(lmr_model.state_dict(), lmr_path)

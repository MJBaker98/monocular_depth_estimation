"""
Script to perform various analysis iterations to generate average performance across iterations
"""

from datetime import datetime

import torch
import torch.nn.functional as F
from sympy.core.random import shuffle
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Subset

from dataloaders.nyu_data import NyuDepthV2
from models.LMR import MaskLearner
from train_dpt import (
    eval,
    init_model,
    train_simple,
    train_with_cutmix,
    train_with_lmr,
)


def main(testing: bool = False):
    """
    Training script
      similar to the train_dpt.py script, but with more epochs for longer training times
    """
    NYU_DATA_PATH = "data/nyu_data/nyu_depth_v2_labeled.mat"

    # download from http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat
    NYU_SPLIT_PATH = "data/nyu_data/splits.mat"

    nyu_test_ds = NyuDepthV2(NYU_DATA_PATH, NYU_SPLIT_PATH, split="test")
    nyu_train_ds = NyuDepthV2(NYU_DATA_PATH, NYU_SPLIT_PATH, split="train")
    nyu_train_dataloader = DataLoader(nyu_train_ds, batch_size=12, shuffle=True)
    nyu_test_dataloader = DataLoader(nyu_test_ds, batch_size=12)

    ### subset dataloaders which can be used to test the various methods
    nyu_single_image_dataset = Subset(nyu_train_ds, [2])
    nyu_single_image_dataloader = DataLoader(nyu_single_image_dataset)

    ########################
    # model training booleans
    do_simple = False
    do_cutmix = False
    do_LMR = True

    ########################
    # Model agnostic hyperparameters
    epochs = 100

    ########################
    # Simple model
    if do_simple:
        simple_model = init_model()

        optim = Adam(simple_model.parameters(), lr=1e-5)
        scheduler = CosineAnnealingLR(optim, eta_min=1e-7, T_max=epochs)

        # standard training - no regularization at all
        simple_log = train_simple(
            model=simple_model,
            loader=nyu_single_image_dataloader,
            optim=optim,
            epochs=epochs,
            scheduler=scheduler,
            save_every=100,
        )
        simple_res = eval(simple_model, nyu_test_dataloader)
        with open(simple_log, "a") as file:
            file.write(f"Eval avg_mse: {simple_res['mse_avg']}")
        timestr = datetime.now().strftime("%a_%d_%b_%Y_%I_%M%p")
        simple_path = "output/checkpoint/simple_model_" + timestr + ".pth"
        torch.save(simple_model.state_dict(), simple_path)

    #######################
    # Cutmix model
    if do_cutmix:
        cutmix_model = init_model()

        optim = Adam(cutmix_model.parameters(), lr=1e-5)
        scheduler = CosineAnnealingLR(optim, eta_min=1e-7, T_max=epochs)

        # standard training - no regularization at all
        cutmix_log = train_with_cutmix(
            model=cutmix_model,
            loader=nyu_train_dataloader,
            optim=optim,
            epochs=epochs,
            cutmix_probability=0.1,
            scheduler=scheduler,
        )
        cutmix_res = eval(cutmix_model, nyu_test_dataloader)
        with open(cutmix_log, "a") as file:
            file.write(f"Eval avg_mse: {cutmix_res['mse_avg']}")
        timestr = datetime.now().strftime("%a_%d_%b_%Y_%I_%M%p")
        cutmix_path = "output/checkpoint/cutmix_model" + timestr + ".pth"
        torch.save(cutmix_model.state_dict(), cutmix_path)

    #######################
    # LMR model
    if do_LMR:
        lmr_model = init_model()
        LMR_model = MaskLearner(device=torch.device("mps"))

        all_params = [
            {"params": lmr_model.parameters(), "lr": 1e-5},
            {"params": LMR_model.parameters(), "lr": 1e-3},
        ]
        optim = Adam(all_params, lr=1e-5)
        # scheduler = CosineAnnealingLR(optim, eta_min=1e-7, T_max=epochs)

        # standard training - no regularization at all
        lmr_log = train_with_lmr(
            model=lmr_model,
            mask_learning_model=LMR_model,
            loader=nyu_single_image_dataloader,
            optim=optim,
            epochs=epochs,
            scheduler=None,
            save_every=100,
            visualize_mask=True,
        )
        if not testing:
            lmr_res = eval(lmr_model, nyu_test_dataloader)
            with open(lmr_log, "a") as file:
                file.write(f"Eval avg_mse: {lmr_res['mse_avg']}")
            timestr = datetime.now().strftime("%a_%d_%b_%Y_%I_%M%p")
            lmr_path = "output/checkpoint/lmr_model" + timestr + ".pth"
            torch.save(lmr_model.state_dict(), lmr_path)


if __name__ == "__main__":
    # train each model for 20 epochs
    main(testing=True)

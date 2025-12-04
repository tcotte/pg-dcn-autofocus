"""
Regression accuracies:
https://machinelearningmastery.com/regression-metrics-for-machine-learning/

Deep learning autofocus:
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8803042/#r24
"""

from __future__ import annotations

import argparse
import os
from typing import Tuple, Dict, Any, List, Union, Optional

import albumentations as A
import imutils.paths
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.base_dataset import AutofocusDatasetFromMetadata
from dataset.preprocessing import filter_by_defocus_sign
from scripts.model import RefocusingNetwork
from utils.logger import WeightandBiaises
from utils.loss import SampleWeightsLoss
from utils.system import get_device, get_os


# -------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------

def move_to_device(
    images: torch.Tensor, labels: torch.Tensor, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Move image and label tensors to specified device."""
    return images.to(device), labels.to(device)


def compute_loss(
    criterion: nn.Module, outputs: torch.Tensor, labels: torch.Tensor, luminance_X: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Compute regression loss, ensuring correct shape."""
    if luminance_X is None:
        return criterion(outputs.squeeze(), labels)
    else:
        return criterion(outputs.squeeze(), labels, luminance_X)


# -------------------------------------------------------------------------
# Training routines
# -------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: Union[torch.device, str],
) -> float:
    """Perform one full training epoch."""
    model.train()
    running_loss = 0.0

    for batch in tqdm(dataloader, desc="Training", unit="batch"):
        images, labels = batch["X"].float(), batch["y"].float()
        images, labels = move_to_device(images, labels, device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = compute_loss(criterion, outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


def evaluate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: Union[torch.device, str],
) -> Tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate one epoch and return avg loss + last batch outputs/images/labels for logging."""
    model.eval()
    running_loss = 0.0

    last_outputs, last_images, last_labels = None, None, None

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", unit="batch"):
            images, labels = batch["X"].float(), batch["y"].float()
            images, labels = move_to_device(images, labels, device)

            outputs = model(images)

            if isinstance(loss, SampleWeightsLoss):
                loss = compute_loss(criterion, outputs, labels, batch['std'].float().to(device))
            else:
                loss = compute_loss(criterion, outputs, labels)

            running_loss += loss.item()

            # store last batch for logging
            last_outputs, last_images, last_labels = outputs, images, labels

    return running_loss / len(dataloader), last_outputs, last_images, last_labels


# -------------------------------------------------------------------------
# Main training logic
# -------------------------------------------------------------------------

def train_regression_model(
    train_dataset: AutofocusDatasetFromMetadata,
    test_dataset: AutofocusDatasetFromMetadata,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    args: argparse.Namespace,
) -> None:

    device = get_device()
    model.to(device)

    config = {
        "device": str(device),
        "loss": str(criterion),
        "optimizer": str(optimizer),
        "lr": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs": args.epoch,
        "dropout": args.dropout,
        "img_size": args.img_size,
        "positive_only": args.positive,
    }

    w_b = WeightandBiaises(
        project_name=args.project_name,
        run_id=args.run_name,
        config=config,
    )

    max_z_value = train_dataset.get_max_defocus()

    for epoch in range(args.epoch):
        print(f"\n----- Epoch {epoch + 1}/{args.epoch} -----")

        # TODO implement We define the focus estimation error DMAE = |Di − ˜Di |

        # ---- TRAIN ----
        train_loss = torch.compile(train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        ), mode='reduce-overhead')

        # ---- VALIDATE ----
        test_loss, last_outputs, last_images, last_labels = torch.compile(evaluate_one_epoch(
            model=model,
            dataloader=test_dataloader,
            criterion=criterion,
            device=device,
        ), mode='reduce-overhead')



        # ---- Log losses ----
        w_b.log_losses(
            train_loss=train_loss,
            test_loss=test_loss,
            epoch=epoch + 1,
        )

        print(f"Train loss: {train_loss:.4f} | Test loss: {test_loss:.4f}")

        if epoch % 10 == 0:
            w_b.save_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                train_loss=train_loss,
                test_loss=test_loss,
            )

            # ---- Log images ----
            if args.normalize_output:
                w_b.log_table(
                    last_outputs.squeeze() * max_z_value,
                    last_images,
                    last_labels * max_z_value,
                    epoch + 1,
                )
            else:
                w_b.log_table(
                    last_outputs.squeeze(),
                    last_images,
                    last_labels,
                    epoch + 1,
                )

    # Save final weights
    torch.save(model.state_dict(), args.run_name + ".pt")
    w_b.save_model(model_name="last.pt", model=model)

    print("\n[SUCCESS] Training finished properly.")


# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:

    # RANDOM SEED
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    np.random.seed(123)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # -------- AUGMENTATIONS --------
    train_transform = A.Compose([
        A.Normalize(),
        A.Resize(args.img_size, args.img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.pytorch.transforms.ToTensorV2(),
    ])

    test_transform = A.Compose([
        A.Normalize(),
        A.Resize(args.img_size, args.img_size),
        A.pytorch.transforms.ToTensorV2(),
    ])

    # -------- DATASETS --------
    X_train = list(imutils.paths.list_images(args.train_set))
    X_test = list(imutils.paths.list_images(args.test_set))

    X_train = filter_by_defocus_sign(X_train, args.positive)
    X_test = filter_by_defocus_sign(X_test, args.positive)

    train_dataset = AutofocusDatasetFromMetadata(X_train, train_transform)
    test_dataset = AutofocusDatasetFromMetadata(X_test, test_transform)

    # -------- DATALOADERS --------
    num_workers = 0 if get_os().lower() == "windows" else os.cpu_count()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    # -------- MODEL / LOSS / OPTIMIZER --------
    model = RefocusingNetwork()
    if args.loss == 0:
        criterion = nn.MSELoss(reduction='mean')
    elif args.loss == 1:
        criterion = nn.L1Loss()
    elif args.loss == 2:
        criterion = nn.SmoothL1Loss()
    elif args.loss == 3:
        criterion = SampleWeightsLoss()
    else:
        raise NotImplementedError('Only L1 / Smooth L1 and SampleWeightsLoss are supported')


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    train_regression_model(
        train_dataset,
        test_dataset,
        train_loader,
        test_loader,
        model,
        optimizer,
        criterion,
        args,
    )


# -------------------------------------------------------------------------
# SCRIPT ENTRYPOINT
# -------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Autofocus on microscope",
        description=(
            "This program enables to train a model which is able to know "
            "from a picture at which distance_af from the focus the microscope is."
        ),
        epilog="--- Tristan COTTE --- SGS France Excellence Opérationnelle ---",
    )

    parser.add_argument(
        "-epoch", "--epoch",
        type=int,
        default=100,
        help="Number of epochs used for train the model",
    )

    parser.add_argument(
        "-device", "--device",
        type=str,
        default="cuda",
        help="Device used to train the model",
    )

    parser.add_argument(
        "-trs", "--train_set",
        type=str,
        help="Dataset of train images",
    )

    parser.add_argument(
        "-tes", "--test_set",
        type=str,
        help="Dataset of test images",
    )

    parser.add_argument(
        "-wd", "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay used to regularized",
    )

    parser.add_argument(
        "-bs", "--batch_size",
        type=int,
        default=64,
        help="Batch size during the training",
    )

    parser.add_argument(
        "-sz", "--img_size",
        type=int,
        default=224,
        help="Training img size",
    )

    parser.add_argument(
        "-do", "--dropout",
        type=float,
        default=0.2,
        help="Dropout used for the training",
    )

    parser.add_argument(
        "-lr", "--learning_rate",
        type=float,
        default=0.0001,
        help="Learning rate used for training",
    )

    parser.add_argument(
        "-project", "--project_name",
        type=str,
        default="Microscope_autofocus",
        help="Name of the project in W&B",
    )

    parser.add_argument(
        "-name", "--run_name",
        type=str,
        default=None,
        help="Name of the run in W&B",
    )

    parser.add_argument(
        "-display", "--interval_display",
        type=int,
        default=10,
        help="Interval of display mask in W&B",
    )

    parser.add_argument(
        "--positive",
        action=argparse.BooleanOptionalAction,
        help="Filter dataset to keep only positive or negative defocus values",
    )

    parser.add_argument(
        "--normalize_output",
        action="store_true",
        help="Normalize regression outputs between 0 and 1",
    )

    parser.add_argument(
        "--loss",
        type=int,
        help="0. L2 / 1. L1 / 2. Smooth L1 / 3. SampleWeightsLoss",
        default=0
    )

    args = parser.parse_args()
    main(args)

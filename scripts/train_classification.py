from __future__ import annotations

import argparse
import os
from typing import Tuple, Dict, Any, List

import albumentations as A
import imutils.paths
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.base_dataset import ClassificationDataset
from dataset.transforms import get_train_transforms, get_valid_transforms
from scripts.model import DCNNetwork
from utils.logger import WeightandBiaises
from utils.metrics import ClassificationMetrics
from utils.system import get_os, get_device, fix_seed, read_yaml_file


# -------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------

def move_batch_to_device(
    images: torch.Tensor, labels: torch.Tensor, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Move tensors to GPU/CPU."""
    return images.to(device), labels.to(device)


def compute_loss(
    criterion: nn.Module, outputs: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Compute loss. Ensures shape consistency."""
    return criterion(outputs, labels)


# -------------------------------------------------------------------------
# Training / Evaluation functions
# -------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    metrics: ClassificationMetrics,
) -> float:
    """Train for one epoch and return avg loss."""

    model.train()
    running_loss = 0.0

    for batch in tqdm(dataloader, unit="batch", desc="Training"):
        images, labels = batch["X"].float(), batch["y"]

        images, labels = move_batch_to_device(images, labels, device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = compute_loss(criterion, outputs, labels.to(torch.float))
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        _, targets = torch.max(labels, 1)

        metrics.update(preds=preds, targets=targets)
        running_loss += loss.item()

    return running_loss / len(dataloader)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    metrics: ClassificationMetrics,
) -> float:
    """Evaluate model; return average loss."""

    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, unit="batch", desc="Evaluating"):
            images, labels = batch["X"].float(), batch["y"]
            images, labels = move_batch_to_device(images, labels, device)

            outputs = model(images)
            loss = compute_loss(criterion, outputs, labels.to(torch.float))

            _, preds = torch.max(outputs, 1)
            _, targets = torch.max(labels, 1)
            metrics.update(preds=preds, targets=targets)

            running_loss += loss.item()

    return running_loss / len(dataloader)


# -------------------------------------------------------------------------
# Main training logic
# -------------------------------------------------------------------------

def train_classification(
    train_dataset: ClassificationDataset,
    test_dataset: ClassificationDataset,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    args: argparse.Namespace,
) -> None:

    device = get_device()

    config = {
        "device": str(device),
        "loss": str(criterion),
        "optimizer": str(optimizer),
        "lr": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "nb_epoch": args.epoch,
        "dropout": args.dropout,
        "img_size": args.img_size,
    }

    w_b = WeightandBiaises(
        project_name=args.project_name,
        run_id=args.run_name,
        config=config,
    )

    model.to(device)

    train_metrics = ClassificationMetrics()
    test_metrics = ClassificationMetrics()

    for epoch in range(args.epoch):
        print(f"\n---- Epoch {epoch + 1}/{args.epoch} ----")

        # Train
        train_metrics.reset()
        train_loss = train_one_epoch(
            model,
            train_dataloader,
            optimizer,
            criterion,
            device,
            train_metrics,
        )

        # Eval
        test_metrics.reset()
        test_loss = evaluate(
            model,
            test_dataloader,
            criterion,
            device,
            test_metrics,
        )

        # Logging
        w_b.log_classification_metrics(
            train_metrics.compute(), set_="train", epoch=epoch + 1
        )
        w_b.log_classification_metrics(
            test_metrics.compute(), set_="test", epoch=epoch + 1
        )
        w_b.log_losses(train_loss=train_loss, test_loss=test_loss, epoch=epoch + 1)

        print(f"Train loss: {train_loss:.4f} | Test loss: {test_loss:.4f}")

        if epoch % args.interval_display == 0:
            w_b.save_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                train_loss=train_loss,
                test_loss=test_loss,
            )

    # Final save
    torch.save(model.state_dict(), args.run_name + ".pt")
    w_b.save_model(model_name="last.pt", model=model)

    print("\n[SUCCESS] Training completed.")


# -------------------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:

    fix_seed()

    configs = read_yaml_file(yaml_file_path=args.configuration_file)


    # Transforms
    train_transform = get_train_transforms(
        normalize=configs['normalization']['used'],
        normalization_mean=configs['normalization']['mean'],
        normalization_std=configs['normalization']['std'])
    test_transform = get_valid_transforms(
        normalize=configs['normalization']['used'],
        normalization_mean=configs['normalization']['mean'],
        normalization_std=configs['normalization']['std'])

    # Datasets
    X_train_images = list(imutils.paths.list_images(args.train_set))
    X_test_images = list(imutils.paths.list_images(args.test_set))

    train_dataset = ClassificationDataset(
        images_list=X_train_images, transform=train_transform
    )
    test_dataset = ClassificationDataset(
        images_list=X_test_images, transform=test_transform
    )

    # Dataloaders
    num_workers = 0 if get_os().lower() == "windows" else os.cpu_count()

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    # Model
    model = DCNNetwork()
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    train_classification(
        train_dataset,
        test_dataset,
        train_dataloader,
        test_dataloader,
        model,
        optimizer,
        criterion,
        args,
    )


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="Autofocus on microscope",
        description=(
            "This program enables training a model able to determine "
            "the distance_af from focus based on an image."
        ),
        epilog="--- Tristan COTTE --- SGS France Excellence Opérationnelle ---",
    )

    parser.add_argument(
        "-epoch", "--epoch", type=int, default=100,
        help="Number of epochs used to train the model",
    )
    parser.add_argument(
        "-device", "--device", type=str, default="cuda",
        help="Device used to train the model",
    )
    parser.add_argument(
        "-trs", "--train_set", type=str, required=False,
        help="Dataset of train images",
    )
    parser.add_argument(
        "-tes", "--test_set", type=str, required=False,
        help="Dataset of test images",
    )
    parser.add_argument(
        "-wd", "--weight_decay", type=float, default=0,
        help="Weight decay used for regularization",
    )
    parser.add_argument(
        "-bs", "--batch_size", type=int, default=64,
        help="Training batch size",
    )
    parser.add_argument(
        "-sz", "--img_size", type=int, default=224,
        help="Training image size",
    )
    parser.add_argument(
        "-do", "--dropout", type=float, default=0.2,
        help="Dropout used for training",
    )
    parser.add_argument(
        "-lr", "--learning_rate", type=float, default=0.0001,
        help="Learning rate",
    )
    parser.add_argument(
        "-project", "--project_name", type=str,
        default="Microscope_autofocus",
        help="Project name for W&B",
    )
    parser.add_argument(
        "-name", "--run_name", type=str, default=None,
        help="Run name in W&B",
    )
    parser.add_argument(
        "-display", "--interval_display", type=int, default=10,
        help="Interval of display mask in W&B and save checkpoint file",
    )

    parser.add_argument(
        "-config", "--configuration_file", type=str, default='config/config.yaml',
        help="Path of configuration file",
    )

    args = parser.parse_args()
    main(args)

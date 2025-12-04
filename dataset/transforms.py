"""
datasets/transforms.py

This module defines image preprocessing and augmentation pipelines
using the Albumentations library, designed for PyTorch-based models.

Reference:
    https://albumentations.ai/docs/
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset.augmentations import SingleColorChannelIsolation, ChangeColorOrder


def get_train_transforms(image_size=224, normalize=True, normalization_mean: tuple[float] = (0.485, 0.456, 0.406),
                         normalization_std: tuple[float] = (0.229, 0.224, 0.225)):
    """
    Returns an Albumentations Compose object for training data.

    Args:
        image_size (int or tuple): Desired output size (H, W).
        normalize (bool): Whether to apply ImageNet normalization.

    Returns:
        albumentations.Compose: Transformation pipeline for training images.
    """

    transforms_list = [
        A.Resize(image_size, image_size),
        A.OneOf(
            transforms=[
                A.RandomBrightnessContrast(p=0.25),
                SingleColorChannelIsolation(p=0.25),
                ChangeColorOrder(p=0.25)
            ],
            p=0.75
        ),
        A.Normalize(),

        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5)
    ]

    if normalize:
        transforms_list.insert(0, A.Normalize(mean=normalization_mean, std=normalization_std))

    transforms_list.append(ToTensorV2())

    return A.Compose(transforms_list)


def get_valid_transforms(image_size=224, normalize=True, normalization_mean: tuple[float] = (0.485, 0.456, 0.406),
                         normalization_std: tuple[float] = (0.229, 0.224, 0.225)):
    """
    Returns an Albumentations Compose object for validation/test data.
    Only deterministic resizing and normalization are applied.

    Args:
        image_size (int or tuple): Desired output size (H, W).
        normalize (bool): Whether to apply ImageNet normalization.

    Returns:
        albumentations.Compose: Transformation pipeline for validation/test images.
    """

    transforms_list = [
        A.Normalize(),
        A.Resize(image_size, image_size),
    ]

    if normalize:
        transforms_list.insert(0, A.Normalize(mean=normalization_mean, std=normalization_std))

    transforms_list.append(ToTensorV2())

    return A.Compose(transforms_list)


def get_inference_transforms(image_size=224, normalize=True, normalization_mean: tuple[float] = (0.485, 0.456, 0.406),
                         normalization_std: tuple[float] = (0.229, 0.224, 0.225)):
    """
    Same as validation transforms, kept separate for clarity and future customization.
    """
    return get_valid_transforms(image_size=image_size, normalize=normalize, normalization_mean=normalization_mean,
                                normalization_std=normalization_std)

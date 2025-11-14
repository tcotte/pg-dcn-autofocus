"""
datasets/transforms.py

This module defines image preprocessing and augmentation pipelines
using the Albumentations library, designed for PyTorch-based models.

Reference:
    https://albumentations.ai/docs/
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size=224, normalize=True):
    """
    Returns an Albumentations Compose object for training data.

    Args:
        image_size (int or tuple): Desired output size (H, W).
        normalize (bool): Whether to apply ImageNet normalization.

    Returns:
        albumentations.Compose: Transformation pipeline for training images.
    """

    transforms_list = [
        # A.Normalize(),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75),
        A.augmentations.geometric.resize.LongestMaxSize(max_size=image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75),
        # A.OneOf([
        #         A.OpticalDistortion(p=0.3),
        #         A.GridDistortion(p=.1)]),
        # A.PixelDropout(dropout_prob=0.01),
        # A.RandomBrightnessContrast(p=0.2),
    ]

    if normalize:
        transforms_list.insert(0, A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))

    transforms_list.append(ToTensorV2())

    return A.Compose(transforms_list)


def get_valid_transforms(image_size=224, normalize=True):
    """
    Returns an Albumentations Compose object for validation/test data.
    Only deterministic resizing and normalization are applied.
    """
    transforms_list = [
        A.augmentations.geometric.resize.LongestMaxSize(max_size=image_size),
    ]

    if normalize:
        transforms_list += [
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225))
        ]

    transforms_list.append(ToTensorV2())

    return A.Compose(transforms_list)


def get_inference_transforms(image_size=224, normalize=True):
    """
    Same as validation transforms, kept separate for clarity and future customization.
    """
    return get_valid_transforms(image_size=image_size, normalize=normalize)

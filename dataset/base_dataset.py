from typing import Union

import exif
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


def get_focus_diff_from_exif_metadata(img_path: str) -> float:
    return float(exif.Image(img_path).make)


class ClassificationDataset(Dataset):
    def __init__(self,
                 images_list: list[str],
                 transform=None) -> None:
        self.images_list = images_list
        self.transform = transform

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        img_path = self.images_list[idx]
        z_value = get_focus_diff_from_exif_metadata(img_path=img_path)

        if z_value <= 0:
            y = torch.tensor((1, 0))

        else:
            y = torch.tensor((0, 1))

        pillow_image = Image.open(img_path)

        if self.transform is None:
            transform = transforms.ToTensor()

            # Convert the image to PyTorch tensor
            tensor_image = transform(pillow_image)

        else:
            transformed = self.transform(image=np.array(pillow_image))
            tensor_image = transformed["image"]

        return {"X": tensor_image, "y": y}


class AutofocusDatasetFromMetadata(Dataset):
    def __init__(self, images_list: list[str], z_range: Union[list, None] = None, normalize_output=False,
                 transform=None):
        if z_range is None:
            z_range = [-np.inf, np.inf]

        self.images_list = images_list
        self.normalize_output = normalize_output

        self.transform = transform

        max_defocus = self.get_max_defocus()
        self.z_range = [-max_defocus, max_defocus]

    def __len__(self):
        return len(self.images_list)

    def get_max_defocus(self) -> float:
        max_value = 0
        for img_path in self.images_list:
            current_value = self.get_focus_diff_from_exif_metadata(img_path=img_path)
            if current_value > max_value:
                max_value = current_value

        return max_value

    @staticmethod
    def get_focus_diff_from_exif_metadata(img_path: str) -> float:
        return float(exif.Image(img_path).make)

    @staticmethod
    def luminance_weighted_std(img_np: np.ndarray) -> float:
        std_image = 0
        # [0.299, 0.587, 0.114] -> RGB luminance weights
        for c, coeff in zip(range(img_np.shape[2]), [0.299, 0.587, 0.114]):
            channel = img_np[..., c]
            std = channel.std()
            std_image += std * coeff

        return std_image

    def __getitem__(self, idx):
        img_path = self.images_list[idx]
        z_value = self.get_focus_diff_from_exif_metadata(img_path=img_path)

        if self.normalize_output:
            z_value = z_value / self.z_range[1]

        pillow_image = Image.open(img_path)

        if self.transform is None:
            transform = transforms.ToTensor()

            # Convert the image to PyTorch tensor
            tensor_image = transform(pillow_image)

        else:
            transformed = self.transform(image=np.array(pillow_image))
            tensor_image = transformed["image"]

        return {"X": tensor_image,
                "y": z_value,
                "image_path": img_path,
                "std": torch.tensor(self.luminance_weighted_std(img_np=np.array(pillow_image)))}

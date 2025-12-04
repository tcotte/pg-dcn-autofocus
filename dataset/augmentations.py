import PIL
import albumentations as A
import cv2
import matplotlib.pyplot as plt
from albumentations.core.transforms_interface import ImageOnlyTransform
import numpy as np
import random
from typing import Any
from imutils.paths import list_images

class SingleColorChannelIsolation(ImageOnlyTransform):
    """
    Very simple color augmentations include isolating
    a single color channel such as R, G, or B. An image can be quickly converted into its
    representation in one color channel by isolating that matrix and adding 2 zero matrices
    from the other color channels.
    Described in paper: A survey on Image Data Augmentation for Deep Learning
    Link: https://link.springer.com/article/10.1186/s40537-019-0197-0
    """

    def __init__(self, p=0.5):
        super().__init__(p=p)
        pass

    def get_params_dependent_on_data(self, params, data):
        isolated_channel = random.randint(0, 2)
        return {"isolated_channel": isolated_channel}

    def apply(self, img, isolated_channel, **params):
        new_img = np.zeros_like(img)

        new_img[:, :, isolated_channel] = img[:, :, isolated_channel]
        return new_img.astype(img.dtype)


class ChangeColorOrder(ImageOnlyTransform):
    """
    Very simple color augmentations include isolating
    a single color channel such as R, G, or B. An image can be quickly converted into its
    representation in one color channel by isolating that matrix and adding 2 zero matrices
    from the other color channels.
    Described in paper: A survey on Image Data Augmentation for Deep Learning
    Link: https://link.springer.com/article/10.1186/s40537-019-0197-0
    """

    def __init__(self, p=0.5):
        super().__init__(p=p)
        self._color_orders = ['RGB', 'GRB', 'GBR']

    def get_params_dependent_on_data(self, params, data):
        color_order = self._color_orders[random.randint(0, 2)]
        return {"color_order": color_order}

    def apply(self, img, color_order, **params):
        if color_order == 'RGB':
            return img
        elif color_order == 'GRB':
            return img[:, :, [1, 0, 2]]
        elif color_order == 'GBR':
            return img[:, :, [1, 2, 0]]
        else:
            raise ValueError('The color_order must be either "RGB", "GRB" or "GBR"')

if __name__ == "__main__":
    # Use it in a pipeline
    pipeline = A.Compose([
        A.Resize(256, 256),
        A.OneOf(
            transforms=[
            A.RandomBrightnessContrast(p=0.25),
            SingleColorChannelIsolation(p=0.25),
            ChangeColorOrder(p=0.25)
            ],
            p=0.75
        ),
        A.Normalize(),

    ])
    list_images = list(list_images(r'D:\03-IDEA\Micronoyaux\Autofocus\11-18-25-dataset\X\train\60310x_22930y'))
    image = PIL.Image.open(list_images[0])

    for j in range(10):
        transformed = pipeline(image=np.array(image))
        tensor_image = transformed["image"]
        plt.imshow(tensor_image)
        plt.show()
import os
from typing import Optional, Union

import cv2
import exif
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from imutils.paths import list_images
from tqdm import tqdm

from dataset.base_dataset import AutofocusDatasetFromMetadata
from scripts.model import DeepCascadeNetwork
from utils.system import get_device

from dataset.transforms import get_valid_transforms
from utils.system import read_yaml_file


def rmse(y_hat, y_ground_truth):
    mse = np.square(np.subtract(np.array(y_ground_truth), np.array(y_hat))).mean()
    return np.sqrt(mse)


def mae(y_hat, y_ground_truth):
    if isinstance(y_hat, list):
        y_hat = np.array(y_hat)

    if isinstance(y_ground_truth, list):
        y_ground_truth = np.array(y_ground_truth)

    return np.abs(y_hat - y_ground_truth).mean()


def load_model(classification_network_weights: str, positive_refocusing_network_weights: str,
               negative_refocusing_network_weights: str, device: str) -> torch.nn.Module:
    model = DeepCascadeNetwork(classification_weights_path=classification_network_weights,
                               positive_weights_path=positive_refocusing_network_weights,
                               negative_weights_path=negative_refocusing_network_weights)

    device = get_device()

    model.to(device)
    model.eval()
    return model


def plot_fov_predictions(
        y,
        y_hat,
        positive_indexes,
        most_focus_gt_image_path: Optional[str],
        most_focus_predicted_image_path: Optional[str],
        min_predicted_defocus: Optional[float]
):
    """Plot FOV predictions with optional ground-truth and predicted images."""

    # ---- Helper: add scatter plot ----
    def add_prediction_scatter(ax):
        ax.set_title(f"RMSE {rmse(y_hat=y_hat, y_ground_truth=y):.2f} / "
                     f"MAE {mae(y_hat=y_hat, y_ground_truth=y):.2f}")
        ax.plot(y, y, label='ground truth')

        # red: positives
        ax.scatter(
            [x for x, i in zip(y, positive_indexes) if i == 1],
            [x for x, i in zip(y_hat, positive_indexes) if i == 1],
            c='r', s=10, label='Predicted positive values'
        )

        # green: negatives
        ax.scatter(
            [x for x, i in zip(y, positive_indexes) if i == 0],
            [x for x, i in zip(y_hat, positive_indexes) if i == 0],
            c='g', s=10, label='Predicted negative values'
        )

        ax.set_xlabel('Z distance_af from focus (µm)')
        ax.set_ylabel('Predicted Z distance_af from focus (µm)')
        ax.legend(loc='upper left')

    # ---- CASE 1: no images ----
    if most_focus_gt_image_path is None:
        fig, ax = plt.subplots(figsize=(20, 20))
        add_prediction_scatter(ax)
        return fig

    # ---- CASE 2: only ground truth image ----
    if most_focus_predicted_image_path is None:
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(40, 20))
        add_prediction_scatter(ax1)
        ax2.imshow(cv2.imread(most_focus_gt_image_path)[:, :, ::-1])
        ax2.set_title("ground truth image")
        ax2.axis("off")
        return fig

    # ---- CASE 3: both images ----
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(60, 20))

    # Prediction plot
    add_prediction_scatter(ax1)

    # GT image
    ax2.imshow(cv2.imread(most_focus_gt_image_path)[:, :, ::-1])
    ax2.set_title("ground truth image")
    ax2.axis("off")

    # Predicted-focus image
    ax3.imshow(cv2.imread(most_focus_predicted_image_path)[:, :, ::-1])
    ax3.set_title(f"Predicted distance from focus {min_predicted_defocus:.2f}")
    ax3.axis("off")

    return fig


def get_most_focus_image_path(path_images: list[str]) -> Optional[str]:
    min_defocus = np.inf
    most_focus_img = None

    for img_path in list(list_images(path_images)):
        img_defocus = float(exif.Image(img_path).make)

        if img_defocus == 0:
            return img_path

        if img_defocus < min_defocus:
            min_defocus = img_defocus
            most_focus_img = img_path

    return most_focus_img


def main(args):
    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)

    device = get_device()

    ### Model
    positive_refocusing_network_weights = args.positive_model_path
    negative_refocusing_network_weights = args.negative_model_path
    classification_network_weights = args.classification_model_path
    model = load_model(classification_network_weights=classification_network_weights,
                       positive_refocusing_network_weights=positive_refocusing_network_weights,
                       negative_refocusing_network_weights=negative_refocusing_network_weights,
                       device=device)

    df_results = pd.DataFrame({
        'XY_position': [],
        'MAE': [],
        'RMSE': [],
        'classification_precision': [],
        'classification_recall': [],
        'classification_f1': []
    })

    path_dataset = os.path.join(args.image_folder)

    for xy_position in os.listdir(path_dataset):
        path_images = os.path.join(path_dataset, xy_position)

        results = predict_fov(model=model,
                              image_folder=path_images,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              device=device,
                              configuration_file=args.configuration_file,
                              coeff=args.multiplier_coeff)
        y_hat = results['refocus_predictions']
        positive_indexes = results['classification_predictions']
        y = results['ground_truth']

        fig = plot_fov_predictions(y, y_hat, positive_indexes, get_most_focus_image_path(path_images),
                                   most_focus_predicted_image_path=results['most_focus_predicted_image_path'],
                                   min_predicted_defocus=results['min_predicted_defocus_gt'])
        fig.savefig(os.path.join(output_folder, f'{xy_position}.jpg'))
        fig.show()

        df_fov = pd.DataFrame({
            'XY_position': [xy_position],
            'MAE': [mae(y_hat=y_hat, y_ground_truth=y)],
            'RMSE': [rmse(y_hat=y_hat, y_ground_truth=y)],
        })
        df_results = pd.concat([df_fov, df_results], ignore_index=True)

    df_results.to_excel(os.path.join(args.output_folder, "results.xlsx"))


def predict_fov(model: torch.nn.Module, image_folder: str, batch_size: int, num_workers: int,
                device: Union[str, torch.device], configuration_file: str,
                coeff: float) -> dict:
    # TODO include dataloader

    configs = read_yaml_file(yaml_file_path=configuration_file)

    # Transforms
    test_transform = get_valid_transforms(
        normalize=configs['normalization']['used'],
        normalization_mean=configs['normalization']['mean'],
        normalization_std=configs['normalization']['std'])

    test_dataset = AutofocusDatasetFromMetadata(images_list=list(list_images(image_folder)),
                                                transform=test_transform)

    y = []
    y_hat = []
    positive_indexes = []
    most_focus_predicted_image_path: Optional[str] = None
    min_predicted_defocus = np.inf

    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset))):
            data = torch.unsqueeze(test_dataset[idx]["X"].float(), dim=0)
            # data_visible = test_dataset[idx]["X"].permute(1, 2, 0)
            data = data.to(device)

            classification, refocus = model(data)
            refocus = refocus.cpu().item()
            cls_prob, cls_prediction = classification
            positive_indexes.append(cls_prediction.cpu().item())

            y_hat.append(refocus*coeff)
            y.append(test_dataset[idx]["y"])

            if np.abs(refocus) < min_predicted_defocus:
                min_predicted_defocus = refocus
                most_focus_predicted_image_path = test_dataset[idx]["image_path"]
                min_predicted_defocus_gt = test_dataset[idx]["y"]

    return {'refocus_predictions': y_hat,
            'classification_predictions': positive_indexes,
            'ground_truth': y,
            'min_predicted_defocus': min_predicted_defocus,
            'most_focus_predicted_image_path': most_focus_predicted_image_path,
            'min_predicted_defocus_gt': min_predicted_defocus_gt
            }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict autofocus values using MobileNetV3 model")

    parser.add_argument(
        "--configuration_file",
        type=str,
        required=True,
        help="Path to configuration file"
    )

    parser.add_argument(
        "--image_folder",
        type=str,
        required=True,
        help="Path to test image folder"
    )

    parser.add_argument(
        "--classification_model_path",
        type=str,
        required=False,
        help="Path to classification model checkpoint"
    )

    parser.add_argument(
        "--negative_model_path",
        type=str,
        required=False,
        help="Path to refocusing negative model checkpoint"
    )

    parser.add_argument(
        "--positive_model_path",
        type=str,
        required=False,
        help="Path to refocusing positmodel checkpoint"
    )

    parser.add_argument(
        "--output_folder",
        type=str,
        default="../output",
        help="Folder to save output prediction plots"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for prediction"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of DataLoader workers"
    )

    parser.add_argument(
        "--multiplier_coeff",
        type=float,
        default=1.0,
        help="Output multiplier: useful when model was trained normalizing outputs"
    )

    args = parser.parse_args()
    main(args)

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import albumentations as A
from torch.utils.data import DataLoader

from dataset.base_dataset import DifferenceAFDataset
from scripts.model import MobileNetV3_Regressor


def rmse(y_hat, y_ground_truth):
    mse = np.square(np.subtract(np.array(y_ground_truth), np.array(y_hat))).mean()
    return np.sqrt(mse)


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model = MobileNetV3_Regressor()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model.eval()


def predict_position(model, df, image_folder, batch_size, num_workers, device):
    test_transform = A.Compose([
        A.Normalize(),
        A.augmentations.geometric.resize.LongestMaxSize(max_size=512),
        A.pytorch.transforms.ToTensorV2()
    ])

    dataset = DifferenceAFDataset(
        df=df,
        image_folder=image_folder,
        transform=test_transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    predictions, targets = [], []

    for batch in dataloader:
        images, labels = batch["X"].float(), batch["y"]
        images = images.to(device)

        with torch.no_grad():
            outputs = model(images)

        batch_targets = labels.cpu().numpy().tolist()
        targets.extend(batch_targets)

        batch_predictions = np.squeeze(outputs.cpu().numpy()).tolist()
        if isinstance(batch_predictions, float):
            predictions.append(batch_predictions)
        else:
            predictions.extend(batch_predictions)

    return np.array(targets), np.array(predictions)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    model = load_model(args.model_path, device)

    # Load dataframe
    df_test = pd.read_excel(args.excel_path)

    # Create output folder if needed
    os.makedirs(args.output_folder, exist_ok=True)

    for position in df_test["xy_position"].unique().tolist():
        sub_df = df_test[df_test["xy_position"] == position]

        targets, predictions = predict_position(
            model=model,
            df=sub_df,
            image_folder=args.image_folder,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device
        )

        error = rmse(predictions, targets)

        # Plot
        fig = plt.figure(figsize=(20, 20))
        plt.title(f"RMSE {error:.2f}")
        plt.suptitle(f"XY position: {position}")
        plt.plot(targets, targets, color="blue", linewidth=1)
        plt.scatter(targets, predictions, color="red")
        plt.xlabel("Z distance_af from focus (µm)")
        plt.ylabel("Predicted Z distance_af from focus (µm)")

        save_path = os.path.join(args.output_folder, f"{position}.jpg")
        plt.savefig(save_path)
        plt.close(fig)

        print(f"Saved plot: {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict autofocus values using MobileNetV3 model")

    parser.add_argument(
        "--image_folder",
        type=str,
        required=True,
        help="Path to test image folder"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )

    parser.add_argument(
        "--excel_path",
        type=str,
        required=True,
        help="Path to Excel file containing test dataset"
    )

    parser.add_argument(
        "--output_folder",
        type=str,
        default="autofocus/output",
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

    args = parser.parse_args()
    main(args)

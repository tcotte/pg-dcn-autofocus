import argparse
import os

import exif
import imutils.paths
import numpy as np
import pandas as pd
from natsort import index_natsorted, order_by_index
from tqdm import tqdm


def create_label_file(set_folder: str, set_name: str, delta: float, output_folder: str = './') -> None:
    df = pd.DataFrame()

    for xy_position in tqdm(os.listdir(set_folder)):
        folder_xy_position = os.path.join(set_folder, xy_position)

        list_distances = []
        list_images = []
        for image in list(imutils.paths.list_images(folder_xy_position)):
            distance_af = float(exif.Image(image).make)
            list_distances.append(round(distance_af, 2))
            list_images.append(image)
            print(f'{os.path.basename(image)} -> {distance_af:.2f}')

        indexes = index_natsorted(list_distances)
        list_distances = np.array(order_by_index(list_distances, indexes))
        list_images = order_by_index(list_images, indexes)

        pair_found = 0
        for distance_af in list_distances[list_distances < -delta]:
            if distance_af + delta in list_distances:
                index_z1 = list_distances.tolist().index(distance_af)
                index_z2 = list_distances.tolist().index(distance_af + delta)

                item_data = {'xy_position': [os.path.basename(folder_xy_position)],
                             'z1_image': [os.path.basename(list_images[index_z1])],
                             'z2_image': [os.path.basename(list_images[index_z2])],
                             'z1_diff_focus': [distance_af],
                             'z2_diff_focus': [distance_af + delta]}
                df = pd.concat([df, pd.DataFrame(data=item_data)], ignore_index=True)
                print(df)

                pair_found += 1

        for distance_af in list_distances[list_distances > 0]:
            if distance_af + delta in list_distances:
                index_z1 = list_distances.tolist().index(distance_af)
                index_z2 = list_distances.tolist().index(distance_af + delta)

                item_data = {'xy_position': [os.path.basename(folder_xy_position)],
                             'z1_image': [os.path.basename(list_images[index_z1])],
                             'z2_image': [os.path.basename(list_images[index_z2])],
                             'z1_diff_focus': [distance_af],
                             'z2_diff_focus': [distance_af + delta]}
                df = pd.concat([df, pd.DataFrame(item_data)], ignore_index=True)

                pair_found += 1

        print(f'pairs found: {pair_found} with a stack of {len(list_distances)} items')

    df.to_excel(os.path.join(output_folder, f'{set_name}.xlsx'))


def main(args: argparse.Namespace) -> None:
    for set_folder, set_name in zip([args.train_dataset_folder, args.test_dataset_folder], ['train', 'test']):
        create_label_file(set_folder=set_folder,
                          set_name=set_name,
                          delta=args.delta,
                          output_folder=args.output_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Label files creator',
        description='The aim of this script is to compute label files suggesting pairs of images I1 and I2.')

    # Add arguments
    parser.add_argument(
        "--delta",
        type=float,
        default=2.0,
        help="Chosen distance between z1 and z2."
    )
    parser.add_argument(
        "--train_dataset_folder",
        type=str,
        required=True,
        help="Train dataset folder"
    )
    parser.add_argument(
        "--test_dataset_folder",
        type=str,
        required=True,
        help="Test dataset folder"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Folder where label files will be created"
    )

    # Parse command line arguments
    args = parser.parse_args()

    # Pass arguments into main()
    main(args)

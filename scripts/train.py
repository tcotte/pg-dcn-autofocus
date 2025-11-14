"""
Regression accuracies: https://machinelearningmastery.com/regression-metrics-for-machine-learning/
Deep learning autofocus: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8803042/#r24
"""
import argparse
import os

import numpy as np
import torch
from datasets.base_dataset import DifferenceAFDataset

from scripts.model import MobileNetV3_Regressor
from utils.logger import WeightandBiaises
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.transforms import get_train_transforms, get_valid_transforms
from utils.system import get_device, get_os
from utils.loss import SampleWeightsLoss


def fix_seed() -> None:
    seed = 123
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train(train_dataloader, test_dataloader, train_dataset, test_dataset, model, optimizer, scheduler, criterion,
          w_b, device) -> None:
    nb_train_batch = int(np.ceil(len(train_dataset) / args.batch_size))
    nb_test_batch = int(np.ceil(len(test_dataset) / args.batch_size))

    mse_func = nn.L1Loss(reduction="sum")

    model.to(device)

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(args.epoch):  # loop over the dataset multiple times
        train_running_loss = 0.0
        test_running_loss = 0.0
        train_mae = 0.0
        test_mae = 0.0

        model.train()

        with tqdm(train_dataloader, unit="batch") as t_epoch:
            for batch in t_epoch:
                t_epoch.set_description(f"Epoch {epoch}")

                # get the inputs; data is a list of [inputs, labels]
                images, labels = batch["X"].float(), batch["y"]

                std = batch['std'].float().to(device)

                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(images)
                train_loss = criterion(outputs.squeeze(), labels)
                train_loss.backward()
                optimizer.step()

                train_mae += mse_func(outputs.squeeze(), labels)

                # print statistics
                train_running_loss += train_loss.item()

        model.eval()
        with torch.no_grad():
            for data in test_dataloader:
                images, labels = data["X"].float(), data["y"]

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                test_mae += mse_func(outputs.squeeze(), labels)
                test_loss = criterion(outputs.squeeze(), labels)

                test_running_loss += test_loss.item()

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        if not args.normalize_output:
            w_b.log_table(outputs.squeeze(), images, labels, epoch + 1)
        else:
            w_b.log_table(outputs.squeeze() * int(args.z_range[1]), images, labels * int(args.z_range[1]),
                          epoch + 1)
            train_mae = train_mae.item() * int(args.z_range[1])
            test_mae = test_mae.item() * int(args.z_range[1])

        train_running_loss = train_running_loss / nb_train_batch
        test_running_loss = test_running_loss / nb_test_batch

        w_b.log_lr(lr=current_lr, epoch=epoch + 1)
        w_b.log_mae(train_mse=train_mae / len(train_dataset), test_mse=test_mae / len(test_dataset),
                    epoch=epoch + 1)
        w_b.log_losses(train_loss=train_running_loss, test_loss=test_running_loss, epoch=epoch + 1)

        print(f"Epoch {str(epoch + 1)}: train_loss {train_running_loss} -- test_loss {test_running_loss} -- "
              f"train_accuracy {train_mae / len(train_dataset)} -- "
              f"test_accuracy {test_mae / len(test_dataset)}")

        train_losses.append(train_running_loss)
        test_losses.append(test_running_loss)
        train_accuracies.append(train_mae / len(train_dataset))
        test_accuracies.append(test_mae / len(test_dataset))

        if epoch % 10 == 0:
            w_b.save_checkpoint(epoch=epoch, model=model, optimizer=optimizer, train_loss=train_running_loss,
                                test_loss=test_running_loss)

    torch.save(model.state_dict(), args.run_name + ".pt")
    w_b.save_model(model_name="last.pt", model=model)

    print("[SUCCESS]: model was trained without disagreement.")


def main(args):
    fix_seed()

    ### Datasets
    # Augmentations
    train_transform = get_train_transforms(image_size=args.img_size, normalize=False)

    test_transform = get_valid_transforms(image_size=args.img_size, normalize=False)

    # Pytorch datasets
    train_path = args.train_set
    test_path = args.test_set

    train_dataset = DifferenceAFDataset.from_excel(excel_filepath=os.path.join(train_path, 'train.xlsx'),
                                                   image_folder=train_path, transform=train_transform)
    test_dataset = DifferenceAFDataset.from_excel(excel_filepath=os.path.join(test_path, 'test.xlsx'),
                                                  image_folder=test_path, transform=test_transform)

    # Dataloaders
    if get_os().lower() == "windows":
        num_workers = 0
    else:
        num_workers = os.cpu_count()

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,
                                 num_workers=num_workers)

    ### Model
    model = MobileNetV3_Regressor(pretrained=args.pretrained_weights, dropout=args.dropout)
    # if not args.lightweight_network:
    #     model = MobileNetV3_Regressor(pretrained=args.pretrained_weights, dropout=args.dropout)
    # else:
    #     model = LightweightNetwork()

    if not args.sample_weights_loss:
        criterion = nn.SmoothL1Loss()

    else:
        criterion = SampleWeightsLoss()

    # Optimizer parameters written in the paper
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    device = get_device()
    conf = {"device": device, "loss": str(criterion), "optimizer": str(optimizer), "lr": args.learning_rate,
            "weight_decay": args.weight_decay, "pretrained_model": args.pretrained_weights,
            "batch_size": args.batch_size, "nb_epoch": args.epoch, "dropout": args.dropout, "img_size": args.img_size}

    w_b = WeightandBiaises(project_name=args.project_name, run_id=args.run_name, config=conf)

    ### Training
    train(train_dataloader, test_dataloader, train_dataset, test_dataset, model, optimizer, scheduler, criterion,
          w_b, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Autofocus on microscope',
        description='This training script enables to train a model following the "Whole slide imaging system using deep'
                    'learning-based automated focusing" paper.',
        epilog='--- Tristan COTTE --- SGS France Innovation Opérationnelle ---')
    parser.add_argument('-epoch', '--epoch', type=int, default=100, required=False,
                        help='Number of epochs used for train the model')
    parser.add_argument('-device', '--device', type=str, default="cuda", required=False,
                        help='Device used to train the model')
    parser.add_argument('-trs', '--train_set', type=str, required=False,
                        help='Dataset of train images')
    parser.add_argument('-tes', '--test_set', type=str, required=False,
                        help='Dataset of test images')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0, required=False,
                        help='Weight decay used to regularized')
    parser.add_argument('-bs', '--batch_size', type=int, default=128, required=False,
                        help='Batch size during the training')
    parser.add_argument('-sz', '--img_size', type=int, default=512, required=False,
                        help='Training img size')
    parser.add_argument('-do', '--dropout', type=float, default=0.2, required=False,
                        help='Dropout used for the training')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, required=False,
                        help='Learning rate used for training')
    parser.add_argument('-project', '--project_name', type=str, default="wsi_autofocus", required=False,
                        help='Name of the project in W&B')
    parser.add_argument('-name', '--run_name', type=str, default=None, required=False,
                        help='Name of the run in W&B')
    parser.add_argument('-display', '--interval_display', type=int, default=10, required=False,
                        help='Interval of display mask in W&B')
    parser.add_argument('-z', '--z_range', nargs='+', help='Picture selection filtered in Z range', required=False)
    parser.add_argument("-weights", "--pretrained_weights", default=False, action="store_true", required=False,
                        help="Use pretrained weights")
    parser.add_argument("--split_by_xy_positions", default=False, action="store_true", required=False,
                        help="Instead of split dataset picture by picture, split the dataset depending on XY position")
    parser.add_argument("-norm", "--normalize_output", default=False, action="store_true", required=False,
                        help="Normalize output in range [-1;1]")
    parser.add_argument("-vit", "--mobile_vit", default=False, action="store_true", required=False,
                        help="Use Mobile Vit instead of MobileNet")
    parser.add_argument("-sw", "--sample_weights_loss", default=False, action="store_true", required=False,
                        help="Use sample weights loss described in WSI system using deep learning-based automated focusing")
    # parser.add_argument("-lwn", "--lightweight_network", default=False, action="store_true", required=False,
    #                     help="Use lightweight network instead of MobileNetv3")

    args = parser.parse_args()

    main(args)

import argparse
import csv
import os
import pathlib
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import onn


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "1", "y"):
        return True
    if value in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def make_complex_field(images):
    return torch.cat(
        (images.unsqueeze(-1), torch.zeros_like(images.unsqueeze(-1))),
        dim=-1,
    ).squeeze(1).contiguous()


def prepare_mnist_images(images):
    images = F.pad(images, pad=(86, 86, 86, 86))
    return make_complex_field(images)


def main(args):
    os.makedirs(args.model_save_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = min(args.num_workers, os.cpu_count() or 1)
    pin_memory = device.type == "cuda"
    print(f"Using device: {device}")

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.MNIST(args.data_path, train=True, transform=transform, download=args.download)
    val_dataset = torchvision.datasets.MNIST(args.data_path, train=False, transform=transform, download=args.download)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=pin_memory,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=pin_memory,
    )

    model = onn.Net().to(device)

    if args.whether_load_model:
        model_path = os.path.join(args.model_save_path, str(args.start_epoch) + args.model_name)
        model.load_state_dict(torch.load(model_path, map_location=device))
        print('Model : "' + model_path + '" loaded.')
    else:
        if os.path.exists(args.result_record_path):
            os.remove(args.result_record_path)
        with open(args.result_record_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Epoch", "Train_Loss", "Train_Acc", "Val_Loss", "Val_Acc", "LR"])

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val_accuracy = 0.0

    for epoch in range(args.start_epoch + 1, args.start_epoch + 1 + args.num_epochs):
        log = [epoch]

        model.train()
        train_len = 0.0
        train_running_counter = 0.0
        train_running_loss = 0.0

        tk0 = tqdm(train_dataloader, ncols=100, total=len(train_dataloader))
        for train_data_batch in tk0:
            train_images = train_data_batch[0].to(device)
            train_labels = train_data_batch[1].to(device)
            train_images = prepare_mnist_images(train_images)

            train_outputs = model(train_images)
            train_loss_ = criterion(train_outputs, train_labels)
            train_counter_ = torch.eq(
                train_labels,
                torch.argmax(train_outputs, dim=1),
            ).float().sum()

            optimizer.zero_grad()
            train_loss_.backward()
            optimizer.step()

            train_len += len(train_labels)
            train_running_loss += train_loss_.item() * len(train_labels)
            train_running_counter += train_counter_.item()

            train_loss = train_running_loss / train_len
            train_accuracy = train_running_counter / train_len

            tk0.set_description_str(
                "Epoch {}/{} : Training".format(epoch, args.start_epoch + args.num_epochs)
            )
            tk0.set_postfix(
                {
                    "Train_Loss": "{:.5f}".format(train_loss),
                    "Train_Accuracy": "{:.5f}".format(train_accuracy),
                }
            )

        log.append(train_loss)
        log.append(train_accuracy)

        with torch.no_grad():
            model.eval()
            val_len = 0.0
            val_running_counter = 0.0
            val_running_loss = 0.0

            tk1 = tqdm(val_dataloader, ncols=100, total=len(val_dataloader))
            for val_data_batch in tk1:
                val_images = val_data_batch[0].to(device)
                val_labels = val_data_batch[1].to(device)
                val_images = prepare_mnist_images(val_images)

                val_outputs = model(val_images)
                val_loss_ = criterion(val_outputs, val_labels)
                val_counter_ = torch.eq(
                    val_labels,
                    torch.argmax(val_outputs, dim=1),
                ).float().sum()

                val_len += len(val_labels)
                val_running_loss += val_loss_.item() * len(val_labels)
                val_running_counter += val_counter_.item()

                val_loss = val_running_loss / val_len
                val_accuracy = val_running_counter / val_len

                tk1.set_description_str(
                    "Epoch {}/{} : Validating".format(epoch, args.start_epoch + args.num_epochs)
                )
                tk1.set_postfix(
                    {
                        "Val_Loss": "{:.5f}".format(val_loss),
                        "Val_Accuracy": "{:.5f}".format(val_accuracy),
                    }
                )

            log.append(val_loss)
            log.append(val_accuracy)

        log.append(args.lr)
        model_path = os.path.join(args.model_save_path, str(epoch) + args.model_name)
        torch.save(model.state_dict(), model_path)
        print('Model : "' + model_path + '" saved.')
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_path = os.path.join(args.model_save_path, "best" + args.model_name)
            torch.save(model.state_dict(), best_model_path)
            print('Best model : "' + best_model_path + '" saved. Val_Accuracy: {:.5f}'.format(best_val_accuracy))

        with open(args.result_record_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-epochs", type=int, default=400)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--data-path", type=str, default="./data", help="path to MNIST data")
    parser.add_argument("--download", type=str2bool, nargs="?", const=True, default=True, help="download MNIST if needed")
    parser.add_argument(
        "--whether-load-model",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="whether to load a saved model",
    )
    parser.add_argument("--start-epoch", type=int, default=0, help="epoch to resume from")
    parser.add_argument("--model-name", type=str, default="_model.pth")
    parser.add_argument("--model-save-path", type=str, default="./saved_model/")
    parser.add_argument(
        "--result-record-path",
        type=pathlib.Path,
        default="./result.csv",
        help="path to save numeric results",
    )

    args_ = parser.parse_args()
    torch.backends.cudnn.benchmark = torch.cuda.is_available()
    random.seed(args_.seed)
    np.random.seed(args_.seed)
    torch.manual_seed(args_.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_.seed)
    main(args_)

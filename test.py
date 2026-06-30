import argparse
import os

import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import onn


def make_complex_field(images):
    return torch.cat(
        (images.unsqueeze(-1), torch.zeros_like(images.unsqueeze(-1))),
        dim=-1,
    ).squeeze(1).contiguous()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/best_model.pth")
    parser.add_argument("--data-path", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--results-dir", type=str, default="./results")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    transform = transforms.Compose([transforms.ToTensor()])

    test_dataset = torchvision.datasets.MNIST(
        root=args.data_path,
        train=False,
        transform=transform,
        download=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = onn.Net().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    correct = 0
    total = 0

    class_correct = [0 for _ in range(10)]
    class_total = [0 for _ in range(10)]

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)

            images = F.pad(images, pad=(86, 86, 86, 86))
            images = make_complex_field(images)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            for label, pred in zip(labels, preds):
                label = label.item()
                pred = pred.item()
                class_total[label] += 1
                if label == pred:
                    class_correct[label] += 1

    print()
    print(f"Test Accuracy: {correct / total * 100:.2f}%")
    print()

    for i in range(10):
        acc = class_correct[i] / class_total[i] * 100
        print(f"Digit {i}: {acc:.2f}%")

    if args.plot:
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

        os.makedirs(args.results_dir, exist_ok=True)
        cm = confusion_matrix(all_labels, all_preds)

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=list(range(10)),
        )

        disp.plot()
        plt.title("MNIST Confusion Matrix")
        output_path = os.path.join(args.results_dir, "confusion_matrix.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print()
        print(f"Confusion matrix saved as {output_path}")


if __name__ == "__main__":
    main()

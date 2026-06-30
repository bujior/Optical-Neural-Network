import argparse

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import onn


def make_complex_field(images):
    return torch.cat(
        (images.unsqueeze(-1), torch.zeros_like(images.unsqueeze(-1))),
        dim=-1,
    ).squeeze(1).contiguous()


def load_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path)
    image = transform(image)

    # Invert white-background images to match MNIST style: white digit on black background.
    if image.mean() > 0.5:
        image = 1.0 - image

    image = image.unsqueeze(0)
    image = F.pad(image, pad=(86, 86, 86, 86))
    image = make_complex_field(image)

    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str, help="path to your handwritten digit image")
    parser.add_argument("--model", type=str, default="models/best_model.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = onn.Net().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    image = load_image(args.image).to(device)

    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred].item() * 100

    print()
    print(f"Prediction: {pred}")
    print(f"Confidence: {confidence:.2f}%")
    print()

    top_probs, top_indices = torch.topk(probs[0], k=3)

    print("Top 3:")
    for idx, prob in zip(top_indices, top_probs):
        print(f"Digit {idx.item()}: {prob.item() * 100:.2f}%")


if __name__ == "__main__":
    main()

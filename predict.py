import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import onn


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CANVAS_SIZE = 28
DIGIT_SIZE = 20
FOREGROUND_THRESHOLD = 0.1


def make_complex_field(images):
    return torch.cat(
        (images.unsqueeze(-1), torch.zeros_like(images.unsqueeze(-1))),
        dim=-1,
    ).squeeze(1).contiguous()


def validate_image_path(image_path):
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
        raise ValueError(f"Unsupported image format: {path.suffix}. Supported formats: {supported}")
    return path


def center_digit(digit):
    _, height, width = digit.shape
    scale = DIGIT_SIZE / max(height, width)
    new_height = max(1, round(height * scale))
    new_width = max(1, round(width * scale))

    resized = transforms.Resize(
        (new_height, new_width),
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )(digit)

    canvas = torch.zeros((1, CANVAS_SIZE, CANVAS_SIZE), dtype=resized.dtype)
    top = (CANVAS_SIZE - new_height) // 2
    left = (CANVAS_SIZE - new_width) // 2
    canvas[:, top:top + new_height, left:left + new_width] = resized
    return canvas


def preprocess_digit_image(image_path):
    path = validate_image_path(image_path)
    image = Image.open(path).convert("L")
    image = transforms.ToTensor()(image)

    # Invert white-background images to match MNIST style: white digit on black background.
    if image.mean() > 0.5:
        image = 1.0 - image

    foreground = image > FOREGROUND_THRESHOLD
    if not foreground.any():
        raise ValueError("No obvious digit was detected in the image. Please use a clearer handwritten digit image.")

    rows = torch.where(foreground[0].any(dim=1))[0]
    cols = torch.where(foreground[0].any(dim=0))[0]
    digit = image[:, rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    return center_digit(digit).clamp(0.0, 1.0)


def save_preprocessed_image(image, output_path):
    output_path = Path(output_path)
    if output_path.parent:
        os.makedirs(output_path.parent, exist_ok=True)
    transforms.ToPILImage()(image).save(output_path)


def load_image(image_path, save_preprocessed=None):
    image = preprocess_digit_image(image_path)
    if save_preprocessed:
        save_preprocessed_image(image, save_preprocessed)

    image = image.unsqueeze(0)
    image = F.pad(image, pad=(86, 86, 86, 86))
    image = make_complex_field(image)

    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str, help="path to your handwritten digit image")
    parser.add_argument("--model", type=str, default="models/best_model.pth")
    parser.add_argument("--save-preprocessed", type=str, default=None, help="optional path to save the 28x28 preprocessed image")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = onn.Net().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    image = load_image(args.image, save_preprocessed=args.save_preprocessed).to(device)

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

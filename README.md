# Optical-Neural-Network

PyTorch version of a Diffractive Deep Neural Network (D2NN) for MNIST digit classification.

Reference paper:

**All optical machine learning using diffractive deep neural networks**

The input image is padded to `200 x 200`, treated as a complex optical field, propagated through diffractive layers, and classified by optical intensity collected from 10 detector regions.

## Project Structure

```text
Optical-Neural-Network/
├── train.py                 # Train the ONN on MNIST
├── test.py                  # Evaluate MNIST accuracy and optionally save confusion matrix
├── predict.py               # Predict your own handwritten digit image
├── onn.py                   # Optical neural network model
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
├── train_autodl.sh          # AutoDL training helper
├── data/                    # MNIST data, ignored by Git
├── models/
│   ├── best_model.pth        # Committed demo/checkpoint model
│   ├── checkpoints/          # Epoch checkpoints, ignored by Git
│   └── training/             # Future training outputs, ignored by Git
├── images/                  # Your own test images, ignored by Git
└── results/                 # Test/prediction outputs, ignored by Git
```

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

If you use GPU, install a PyTorch build that matches your CUDA environment.

## Training

Linux / AutoDL:

```bash
python train.py \
  --data-path ./data \
  --model-save-path ./models/checkpoints \
  --best-model-path ./models/training/best_model.pth \
  --result-record-path ./results/result.csv \
  --batch-size 1024 \
  --num-epochs 120 \
  --num-workers 8 \
  --lr 0.001
```

Windows PowerShell:

```powershell
python train.py `
  --data-path ./data `
  --model-save-path ./models/checkpoints `
  --best-model-path ./models/training/best_model.pth `
  --result-record-path ./results/result.csv `
  --batch-size 1024 `
  --num-epochs 120 `
  --num-workers 8 `
  --lr 0.001
```

Training uses CUDA automatically if available; otherwise it falls back to CPU.

Training outputs:

- `models/checkpoints/{epoch}_model.pth`: checkpoint saved after each epoch.
- `models/training/best_model.pth`: best model from future training runs.
- `results/result.csv`: epoch loss, accuracy, and learning rate.

Future training outputs are ignored by Git. The committed model for demonstration and reproduction is:

```text
models/best_model.pth
```

## Resume Training

For example, to resume from `models/checkpoints/120_model.pth`:

```bash
python train.py \
  --whether-load-model true \
  --start-epoch 120 \
  --model-save-path ./models/checkpoints \
  --model-name _model.pth
```

The loaded checkpoint path is:

```text
{model-save-path}/{start-epoch}{model-name}
```

## Test on MNIST

Evaluate the committed best model:

```bash
python test.py --model models/best_model.pth --batch-size 512
```

Save a confusion matrix:

```bash
python test.py --model models/best_model.pth --plot
```

The confusion matrix is saved to:

```text
results/confusion_matrix.png
```

## Predict Your Own Handwritten Image

Put your image under `images/`, for example:

```text
images/7.jpg
```

Run prediction:

```powershell
python predict.py images\7.jpg
```

Linux / AutoDL:

```bash
python predict.py images/7.jpg
```

By default, `predict.py` loads:

```text
models/best_model.pth
```

To use another model:

```powershell
python predict.py images\7.jpg --model models\training\best_model.pth
```

Supported image formats:

- `.jpg`
- `.jpeg`
- `.png`

`predict.py` automatically preprocesses your image:

- converts it to grayscale;
- inverts white-background images to MNIST style;
- crops the digit region;
- resizes it while keeping aspect ratio;
- centers it on a `28 x 28` canvas;
- pads it to `200 x 200` before sending it into the ONN.

If prediction looks wrong, save the preprocessed image:

```powershell
python predict.py images\7.jpg --save-preprocessed results\7_preprocessed.png
```

The saved file shows the actual `28 x 28` image seen by the model. Use it to check whether the digit is too small, off-center, too thin, or visually different from MNIST.

## AutoDL

On AutoDL, run:

```bash
bash train_autodl.sh
```

The script assumes the project path is:

```text
/root/autodl-tmp/Optical-Neural-Network
```

If your project is in another directory, edit the `cd` line in `train_autodl.sh`.

## Git Ignore Policy

The following files are intentionally not committed:

- `data/`: MNIST data.
- `images/*`: your personal handwritten test images.
- `results/*`: generated reports, plots, and preprocessed images.
- `models/checkpoints/*`: epoch checkpoints.
- `models/training/*`: future training outputs.
- `__pycache__/` and `*.pyc`: Python caches.

The repository keeps `models/best_model.pth` as a committed demonstration checkpoint so that prediction and testing work immediately after cloning.

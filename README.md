# Optical-Neural-Network

这是一个用 PyTorch 实现的 Diffractive Deep Neural Network (D2NN) 小项目，用 MNIST 做手写数字分类。

参考论文：

**All optical machine learning using diffractive deep neural networks**

输入图像会先补到 `200 x 200`，作为复数光场进入多层衍射网络，最后用 10 个 detector region 的光强进行分类。

## 目录

```text
Optical-Neural-Network/
|-- train.py                 # 训练
|-- test.py                  # 测试 MNIST 准确率，可生成混淆矩阵
|-- predict.py               # 识别自己的手写图片
|-- onn.py                   # ONN 模型
|-- requirements.txt         # 依赖
|-- README.md
|-- train_autodl.sh          # AutoDL 启动脚本
|-- data/                    # MNIST 数据，不提交 GitHub
|-- models/
|   |-- best_model.pth        # 当前展示用模型，已提交
|   |-- checkpoints/          # 每轮 checkpoint，不提交 GitHub
|   `-- training/             # 后续训练输出，不提交 GitHub
|-- images/                  # 自己测试的图片，不提交 GitHub
`-- results/                 # 测试和预测输出，不提交 GitHub
```

## 安装

```bash
pip install -r requirements.txt
```

如果使用 GPU，建议按自己机器的 CUDA 版本安装对应的 PyTorch。

## 训练

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

训练会自动检测 CUDA；没有 CUDA 就用 CPU。

默认输出：

- `models/checkpoints/{epoch}_model.pth`：每轮 checkpoint。
- `models/training/best_model.pth`：后续训练得到的 best model。
- `results/result.csv`：每轮 loss、accuracy 和学习率。

这些训练输出默认不提交 GitHub。

## 继续训练

例如从 `models/checkpoints/120_model.pth` 继续：

```bash
python train.py \
  --whether-load-model true \
  --start-epoch 120 \
  --model-save-path ./models/checkpoints \
  --model-name _model.pth
```

实际加载路径是：

```text
{model-save-path}/{start-epoch}{model-name}
```

## 测试 MNIST

使用当前展示模型测试：

```bash
python test.py --model models/best_model.pth --batch-size 512
```

生成混淆矩阵：

```bash
python test.py --model models/best_model.pth --plot
```

输出位置：

```text
results/confusion_matrix.png
```

## 预测自己的图片

把图片放到 `images/`，例如：

```text
images/7.jpg
```

运行：

```powershell
python predict.py images\7.jpg
```

Linux / AutoDL:

```bash
python predict.py images/7.jpg
```

默认模型：

```text
models/best_model.pth
```

如果要换模型：

```powershell
python predict.py images\7.jpg --model models\training\best_model.pth
```

支持图片格式：

- `.jpg`
- `.jpeg`
- `.png`

`predict.py` 会自动做预处理：

- 转灰度；
- 白底黑字自动反色为 MNIST 风格；
- 裁剪数字区域；
- 保持比例缩放；
- 居中到 `28 x 28`；
- 再补到 `200 x 200` 输入 ONN。

如果识别结果不对，可以保存模型实际看到的预处理图：

```powershell
python predict.py images\7.jpg --save-preprocessed results\7_preprocessed.png
```

打开 `results/7_preprocessed.png`，看数字是否太小、偏移、太细，或者不像 MNIST。

## AutoDL

在 AutoDL 上：

```bash
bash train_autodl.sh
```

脚本默认项目路径：

```text
/root/autodl-tmp/Optical-Neural-Network
```

如果项目放在其他位置，改 `train_autodl.sh` 里的 `cd` 路径。

## Git 忽略规则

不会提交：

- `data/`
- `images/*`
- `results/*`
- `models/checkpoints/*`
- `models/training/*`
- `__pycache__/`
- `*.pyc`

保留提交：

- `models/best_model.pth`

这个模型用于快速测试 `test.py` 和 `predict.py`。

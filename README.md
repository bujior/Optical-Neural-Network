# Optical-Neural-Network

PyTorch 版的 Diffractive Deep Neural Network 小复现，参考论文：

**All optical machine learning using diffractive deep neural networks**

代码用 MNIST 做分类实验。输入图片先补到 `200 x 200`，再当作复数光场送进多层衍射网络，最后用 10 个 detector region 的光强做分类。

这个仓库主要是方便自己跑通流程和继续改实验，不包含已经训练好的权重。

## 文件

```text
.
├── onn.py              # 网络结构
├── train.py            # 训练脚本
├── train_autodl.sh     # AutoDL 上用的启动脚本
├── requirements.txt    # 依赖
├── tests/              # 简单测试
└── README.md
```

## 安装

先装依赖：

```bash
pip install -r requirements.txt
```

如果要用 GPU，PyTorch 最好按自己机器的 CUDA 版本单独安装，别直接套一个固定版本。

## 训练

Linux / macOS：

```bash
python train.py \
  --data-path ./data \
  --model-save-path ./saved_model \
  --result-record-path ./result.csv \
  --batch-size 1024 \
  --num-epochs 120 \
  --num-workers 8 \
  --lr 0.001
```

Windows PowerShell：

```powershell
python train.py `
  --data-path ./data `
  --model-save-path ./saved_model `
  --result-record-path ./result.csv `
  --batch-size 1024 `
  --num-epochs 120 `
  --num-workers 8 `
  --lr 0.001
```

脚本会自动用 CUDA；没有 CUDA 就用 CPU。完整训练比较慢，第一次可以先把 `--num-epochs` 调小，确认环境没问题。

## 输出

训练会生成：

- `saved_model/{epoch}_model.pth`：每轮保存一次。
- `saved_model/best_model.pth`：验证集准确率最高的一次。
- `result.csv`：每轮的 loss、accuracy 和学习率。

这些文件已经放进 `.gitignore`，本地训练出来后不会默认提交。

## 继续训练

比如已经有 `saved_model/120_model.pth`，可以这样接着跑：

```bash
python train.py \
  --whether-load-model true \
  --start-epoch 120 \
  --model-save-path ./saved_model \
  --model-name _model.pth
```

实际加载的文件路径是：

```text
{model-save-path}/{start-epoch}{model-name}
```

上面的例子就是：

```text
./saved_model/120_model.pth
```

## AutoDL

AutoDL 上可以直接用：

```bash
bash train_autodl.sh
```

脚本里默认项目路径是：

```text
/root/autodl-tmp/Optical-Neural-Network
```

如果你放在别的目录，改一下 `train_autodl.sh` 里的 `cd` 和输出路径就行。

## 测试

语法检查：

```bash
python -m py_compile onn.py train.py
```

简单前向测试：

```bash
python -m unittest discover -s tests
```

这个测试不会下载 MNIST，也不会训练模型，只检查 `onn.Net` 能不能正常前向输出 `(batch, 10)`。

## 一些说明

- MNIST 原图是 `28 x 28`，训练前会补零到 `200 x 200`。
- 网络输出不是传统神经网络的 logits，而是 detector region 光强取对数后的结果。
- 这个实现主要是复现实验流程，不等同于论文里的真实光学硬件设置。
- `requirements.txt` 不锁死版本，主要是因为不同机器的 CUDA / PyTorch 安装方式差别比较大。

## 这次整理改了什么

- 重新写了 README，去掉原来的乱码，补了训练、恢复训练、AutoDL 和测试命令。
- 加了 `.gitignore`，忽略缓存、数据集、模型权重和训练结果。
- 加了 `tests/test_onn.py`，用一个很小的 smoke test 检查模型前向传播。
- 清理了误生成的 `__pycache__/`。
- 把 `train.py` 里重复的 MNIST 预处理抽成了 `prepare_mnist_images()`，训练逻辑和参数不变。

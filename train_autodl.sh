#!/usr/bin/env bash
# AutoDL 训练启动脚本：用于在远程 GPU 环境中一键运行 train.py。
# 该脚本只封装训练命令和路径参数，不包含模型结构、数据集或模型权重。
set -e

cd /root/autodl-tmp/Optical-Neural-Network

python train.py \
  --data-path /root/autodl-tmp/data \
  --model-save-path /root/autodl-tmp/onn_saved_model \
  --result-record-path /root/autodl-tmp/onn_result.csv \
  --batch-size 1024 \
  --num-epochs 120 \
  --num-workers 8 \
  --lr 0.001

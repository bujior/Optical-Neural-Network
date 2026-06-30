#!/usr/bin/env bash
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

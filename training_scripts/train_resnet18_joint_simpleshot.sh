#!/usr/bin/env bash

# add GPU as argument (just the number) e.g. "train_feat_modelnet.sh 0"
export CUDA_VISIBLE_DEVICES=$1

echo "Using GPU $1"

# train modelnet
python lssb/lowshot/train.py --cfg=lssb/lowshot/configs/simpleshot/modelnet/modelnet-joint-simpleshot-resnet18-cfg.yaml;
# train shapenet
python lssb/lowshot/train.py --cfg=lssb/lowshot/configs/simpleshot/shapenet/shapenet-joint-simpleshot-resnet18-cfg.yaml;
# train toys4k
python lssb/lowshot/train.py --cfg=lssb/lowshot/configs/simpleshot/toys/TOYS4K-joint-simpleshot-resnet18-cfg.yaml;

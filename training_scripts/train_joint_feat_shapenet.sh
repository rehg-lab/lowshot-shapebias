#!/usr/bin/env bash

# add GPU as argument (just the number) e.g. "train_feat_modelnet.sh 0"
export CUDA_VISIBLE_DEVICES=1

python lssb/lowshot/train.py --cfg=lssb/lowshot/configs/feat/shapenet/shapenet-joint-feat-resnet18-1-shot-5-way-cfg.yaml;

python lssb/lowshot/train.py --cfg=lssb/lowshot/configs/feat/shapenet/shapenet-joint-feat-resnet18-5-shot-5-way-cfg.yaml;

python lssb/lowshot/train.py --cfg=lssb/lowshot/configs/feat/shapenet/shapenet-joint-feat-resnet18-1-shot-10-way-cfg.yaml;

python lssb/lowshot/train.py --cfg=lssb/lowshot/configs/feat/shapenet/shapenet-joint-feat-resnet18-5-shot-10-way-cfg.yaml;

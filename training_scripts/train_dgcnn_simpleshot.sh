#!/usr/bin/env bash

# requires 3 gpus for default batch size
export CUDA_VISIBLE_DEVICES=0,1,2

# train modelnet
python lssb/lowshot/train.py --cfg=lssb/lowshot/configs/simpleshot/modelnet/modelnet-simpleshot-dgcnn-cfg.yaml;
# train shapenet
python lssb/lowshot/train.py --cfg=lssb/lowshot/configs/simpleshot/shapenet/shapenet-simpleshot-dgcnn-cfg.yaml;
# train toys4k
python lssb/lowshot/train.py --cfg=lssb/lowshot/configs/simpleshot/toys/TOYS4K-simpleshot-dgcnn-cfg.yaml;

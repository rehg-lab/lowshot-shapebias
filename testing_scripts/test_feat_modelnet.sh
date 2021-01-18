#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

python lssb/lowshot/test.py --log_dir=pretrained_models/feat/modelnet/image-only \
                            --name=modelnet-feat-1-shot-5-way-resnet18 \
                            --version=0 \
                            --gpu=1;

python lssb/lowshot/test.py --log_dir=pretrained_models/feat/modelnet/image-only \
                            --name=modelnet-feat-5-shot-5-way-resnet18 \
                            --version=0 \
                            --gpu=1;

python lssb/lowshot/test.py --log_dir=pretrained_models/feat/modelnet/image-only \
                            --name=modelnet-feat-1-shot-10-way-resnet18 \
                            --version=0 \
                            --gpu=1;

python lssb/lowshot/test.py --log_dir=pretrained_models/feat/modelnet/image-only \
                            --name=modelnet-feat-5-shot-10-way-resnet18 \
                            --version=0 \
                            --gpu=1;


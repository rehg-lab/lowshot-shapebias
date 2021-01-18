#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2

python lssb/lowshot/test.py --log_dir=pretrained_models/feat/modelnet/shape-biased \
                            --name=modelnet-joint-feat-1-shot-5-way-resnet18 \
                            --version=0 \
                            --gpu=1;

#python lssb/lowshot/test.py --log_dir=pretrained_models/feat/modelnet/shape-biased \
#                            --name=modelnet-joint-feat-5-shot-5-way-resnet18 \
#                            --version=0
#                            --gpu=1;

#python lssb/lowshot/test.py --log_dir=pretrained_models/feat/modelnet/shape-biased \
#                            --name=modelnet-joint-feat-1-shot-10-way-resnet18 \
#                            --version=0 \
#                            --gpu=1;

#python lssb/lowshot/test.py --log_dir=pretrained_models/feat/modelnet/shape-biased \
#                            --name=modelnet-joint-feat-5-shot-10-way-resnet18 \ 
#                            --version=0 \
#                            --gpu=1;

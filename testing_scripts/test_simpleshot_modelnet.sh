#!/usr/bin/env bash

#export CUDA_VISIBLE_DEVICES=0

python lssb/lowshot/test.py --log_dir=pretrained_models/simpleshot/modelnet/image-only/ \
                            --name=resnet18-modelnet-simpleshot \
                            --version=0 \
                            --gpu=1 \

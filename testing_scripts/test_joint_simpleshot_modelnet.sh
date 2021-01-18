#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1

python lssb/lowshot/test.py --log_dir=pretrained_models/simpleshot/modelnet/shape-biased/ \
                            --name=joint-modelnet-pairwise-simpleshot \
                            --version=0 \
                            --gpu=1 \

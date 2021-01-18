#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

python lssb/lowshot/test.py --log_dir=pretrained_models/simpleshot/shapenet/shape-biased/ \
                            --name=joint-shapenet-pairwise-simpleshot \
                            --version=0 \
                            --gpu=1 \

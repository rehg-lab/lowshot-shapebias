#!/bin/bash

data_path='<path to shapenet objs>'
param_path='./data_generation_parameters.json'
output_path='<output path>'
blender_path='<blender path>'

python wrapper.py \
    --start=0 \
    --end=55000 \
    --gpu_idx=0 \
    --input_path=$data_path \
    --output_path=$output_path \
    --blender_path=$blender_path \
    --param_path=$param_path \
    --dataset_type=shapenet 2>&1 | tee datagen_log_shapenet.txt

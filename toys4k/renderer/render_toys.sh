#!/bin/bash

data_path='<path to toys>'
param_path='./data_generation_parameters.json'
output_path='<output path>'
blender_path='<blender path>'

python wrapper.py \
    --start=0 \
    --end=4010 \
    --gpu_idx=0 \
    --input_path=$data_path \
    --output_path=$output_path \
    --blender_path=$blender_path \
    --param_path=$param_path \
    --dataset_type=toys 2>&1 | tee toys_datagen_logs.txt


#!/bin/bash

data_path='./demo_assets/'
output_path='./demo_output'
param_path='./demo_assets/demo_data_generation_parameters.json'
blender_path='<path to blender>'

# gpu_idx=0 will use the GPU corresponding to what's available in nvidia-smi 
# if no GPU is available, set scn.cycles.device = "CPU" in L40 of generate.py
# and uncomment L40-48

python wrapper.py \
    --start=0 \
    --end=1 \
    --gpu_idx=0 \
    --demo=1 \
    --input_path=$data_path \
    --output_path=$output_path \
    --param_path=$param_path \
    --blender_path=$blender_path \
    --dataset_type=toys 2>&1 | tee demo_datagen_log.txt

python ../util_scripts/overlay.py --data_path=$output_path


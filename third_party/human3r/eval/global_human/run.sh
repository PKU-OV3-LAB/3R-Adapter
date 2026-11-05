#!/bin/bash

set -e

workdir='.'
model_name='human3r'
ckpt_name='human3r'
model_weights="${workdir}/src/${ckpt_name}.pth"
datasets=('3dpw' 'emdb1')

for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/global_human/${data}_${model_name}"
    echo "$output_dir"
    accelerate launch --num_processes 1 --main_process_port 29550 eval/global_human/launch.py \
        --weights "$model_weights" \
        --output_dir "$output_dir" \
        --eval_dataset "$data" \
        --size 512 \
        --reset_interval 100 \
        --use_ttt3r
        # --save
        # --vis
done



#!/bin/bash
set -e
trap 'trap - SIGTERM && kill -- -$$' SIGINT SIGTERM
exp_name=$1
GPUs=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
sort -nk 2 -r | awk '$2>3000 {print $1}' | tr -d "\n")
IFS="," read -r -a GPUs <<< "$GPUs"
num_GPUs=${#GPUs[@]}
GPU_index=0
if [ "$num_GPUs" -eq 0 ]; then
  echo "No enough GPU memory"
  exit 0
fi
resample_splits=(1 2 4)
sigmas=(0.01 0.1 1.0)
draw_load=()
for resample_split in "${resample_splits[@]}"; do
  for sigma in "${sigmas[@]}"; do
    draw_load=("${draw_load[@]}" "LWIS_${resample_split}_${sigma}")
  done
done
CUDA_VISIBLE_DEVICES=${GPUs[GPU_index]} DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py \
--load "${draw_load[@]}" --sensitivity &> experiments/"$exp_name"/draw_sensitivity.txt &
GPU_index=$(((GPU_index+1)%num_GPUs))

#!/bin/bash
set -e
exp_name=$1
GPUs=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
sort -nk 2 -r | awk '$2>3000 {print $1}' | tr -d "\n")
IFS="," read -r -a GPUs <<< "$GPUs"
num_GPUs=${#GPUs[@]}
GPU_index=0
if [ $num_GPUs -eq 0 ]; then
  echo "No enough GPU memory"
  exit 0
fi
if [ "$exp_name" == "navier_stokes" ]; then
  CUDA_VISIBLE_DEVICES=${GPUs[GPU_index]} DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py \
  --load PINN_10.0 LWIS_10.0 --re 10 &> experiments/"$exp_name"/draw_10.txt &
  GPU_index=$(((GPU_index+1)%num_GPUs))
  CUDA_VISIBLE_DEVICES=${GPUs[GPU_index]} DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py \
  --load PINN_100.0 LWIS_100.0 --re 100 &> experiments/"$exp_name"/draw_100.txt &
  GPU_index=$(((GPU_index+1)%num_GPUs))
  CUDA_VISIBLE_DEVICES=${GPUs[GPU_index]} DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py \
  --load PINN_1000.0 LWIS_1000.0 --re 1000 &> experiments/"$exp_name"/draw_1000.txt &
  GPU_index=$(((GPU_index+1)%num_GPUs))
elif [ "$exp_name" == "schrodinger" ]; then
  num_train_samples_domain=10000
  resample_times=5
  resample_numbers=1000
  data_multipliers=(1 4 16)
  sigmas=(0.05 0.1 0.2)
  draw_load=()
  for data_multiplier in "${data_multipliers[@]}"; do
    num_train_samples=$((data_multiplier*(num_train_samples_domain+resample_times*resample_numbers)))
    draw_load=("${draw_load[@]}" "PINN_${num_train_samples}")
  done
  for data_multiplier in "${data_multipliers[@]}"; do
    num_train_samples=$((data_multiplier*(num_train_samples_domain+resample_times*resample_numbers)))
    for sigma in "${sigmas[@]}"; do
      draw_load=("${draw_load[@]}" "LWIS_${num_train_samples}_${sigma}")
    done
  done
  CUDA_VISIBLE_DEVICES=${GPUs[GPU_index]} DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py \
  --load "${draw_load[@]}" &> experiments/"$exp_name"/draw_sensitivity.txt &
  GPU_index=$(((GPU_index+1)%num_GPUs))
else
  CUDA_VISIBLE_DEVICES=${GPUs[GPU_index]} DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py \
  --load PINN LWIS &> experiments/"$exp_name"/draw.txt &
  GPU_index=$(((GPU_index+1)%num_GPUs))
fi

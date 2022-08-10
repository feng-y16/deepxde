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
if [ "$exp_name" == "navier_stokes" ]; then
  res=(10 100)
  for re in "${res[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPUs[GPU_index]} DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py \
    --load PINN_"$re".0 LWIS-D_"$re".0 LWIS-B_"$re".0 LWIS_"$re".0 --re "$re" --draw-annealing \
    &> experiments/"$exp_name"/draw_ablation_"$re".0.txt &
    GPU_index=$(((GPU_index+1)%num_GPUs))
  done
else
  CUDA_VISIBLE_DEVICES=${GPUs[GPU_index]} DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py \
  --load PINN LWIS-D LWIS-B LWIS --draw-annealing &> experiments/"$exp_name"/draw_ablation.txt &
  GPU_index=$(((GPU_index+1)%num_GPUs))
fi

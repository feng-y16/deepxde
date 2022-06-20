#!/bin/bash
set -e
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
bash clean.sh "$exp_name"
if [ "$exp_name" == "navier_stokes" ]; then
  CUDA_VISIBLE_DEVICES=${GPUs[GPU_index]} DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py \
  --re 10 --num-train-samples-domain 500 --resample-numbers 500 --num-test-samples 512 \
  &> experiments/"$exp_name"/PINN_10.0.txt &
  GPU_index=$(((GPU_index+1)%num_GPUs))
  CUDA_VISIBLE_DEVICES=${GPUs[GPU_index]} DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py \
  --resample --re 10 --num-train-samples-domain 500 --resample-numbers 500 --num-test-samples 512 \
  &> experiments/"$exp_name"/LWIS_10.0.txt &
  GPU_index=$(((GPU_index+1)%num_GPUs))
  CUDA_VISIBLE_DEVICES=${GPUs[GPU_index]} DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py \
  --re 100 --num-train-samples-domain 5000 --resample-numbers 5000 --num-test-samples 512 \
  &> experiments/"$exp_name"/PINN_100.0.txt &
  GPU_index=$(((GPU_index+1)%num_GPUs))
  CUDA_VISIBLE_DEVICES=${GPUs[GPU_index]} DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py \
  --resample --re 100 --num-train-samples-domain 5000 --resample-numbers 5000 --num-test-samples 512 \
  &> experiments/"$exp_name"/LWIS_100.0.txt &
  GPU_index=$(((GPU_index+1)%num_GPUs))
  CUDA_VISIBLE_DEVICES=${GPUs[GPU_index]} DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py \
  --re 1000 --num-train-samples-domain 50000 --resample-numbers 50000 --num-test-samples 512 \
  &> experiments/"$exp_name"/PINN_1000.0.txt &
  GPU_index=$(((GPU_index+1)%num_GPUs))
  CUDA_VISIBLE_DEVICES=${GPUs[GPU_index]} DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py \
  --resample --re 1000 --num-train-samples-domain 50000 --resample-numbers 50000 --num-test-samples 512 \
  &> experiments/"$exp_name"/LWIS_1000.0.txt &
  GPU_index=$(((GPU_index+1)%num_GPUs))
elif [ "$exp_name" == "schrodinger" ]; then
  bash run_sensitivity.sh
  exit 0
else
  CUDA_VISIBLE_DEVICES=${GPUs[GPU_index]} DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py \
  &> experiments/"$exp_name"/PINN.txt &
  GPU_index=$(((GPU_index+1)%num_GPUs))
  CUDA_VISIBLE_DEVICES=${GPUs[GPU_index]} DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py \
  --resample &> experiments/"$exp_name"/LWIS.txt &
  GPU_index=$(((GPU_index+1)%num_GPUs))
fi
set +e
num_jobs=$(jobs | grep -c "")
while [ "$num_jobs" -ge 1 ]
do
  num_jobs=$(jobs | grep -c "Run")
  echo "$exp_name" "$num_jobs" "jobs remaining"
  sleep 2
done
set -e
./draw.sh "$exp_name"
echo "$exp_name" "complete"

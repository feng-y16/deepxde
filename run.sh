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
bash clean.sh "$exp_name"
if [ "$exp_name" == "navier_stokes" ]; then
  res=(10 100)
  num_train_samples_domain_per_re=500
  num_train_samples_boundary_per_re=50
  num_train_samples_initial_per_re=50
  for re in "${res[@]}"; do
    num_train_samples_domain=$((re*num_train_samples_domain_per_re))
    num_train_samples_boundary=$((re*num_train_samples_boundary_per_re))
    num_train_samples_initial=$((re*num_train_samples_initial_per_re))
    CUDA_VISIBLE_DEVICES=${GPUs[GPU_index]} DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py \
    --num-train-samples-domain $num_train_samples_domain \
    --num-train-samples-boundary $num_train_samples_boundary \
    --num-train-samples-initial $num_train_samples_initial \
    --re "$re" &> experiments/"$exp_name"/PINN_"$re".0.txt &
    GPU_index=$(((GPU_index+1)%num_GPUs))
    CUDA_VISIBLE_DEVICES=${GPUs[GPU_index]} DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py \
    --num-train-samples-domain $num_train_samples_domain \
    --num-train-samples-boundary $num_train_samples_boundary \
    --num-train-samples-initial $num_train_samples_initial \
    --re "$re" --annealing &> experiments/"$exp_name"/PINN-A_"$re".0.txt &
    GPU_index=$(((GPU_index+1)%num_GPUs))
    CUDA_VISIBLE_DEVICES=${GPUs[GPU_index]} DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py \
    --num-train-samples-domain $num_train_samples_domain \
    --num-train-samples-boundary $num_train_samples_boundary \
    --num-train-samples-initial $num_train_samples_initial \
    --re "$re" --resample &> experiments/"$exp_name"/LWIS_"$re".0.txt &
    GPU_index=$(((GPU_index+1)%num_GPUs))
    CUDA_VISIBLE_DEVICES=${GPUs[GPU_index]} DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py \
    --num-train-samples-domain $num_train_samples_domain \
    --num-train-samples-boundary $num_train_samples_boundary \
    --num-train-samples-initial $num_train_samples_initial \
    --re "$re" --adversarial &> experiments/"$exp_name"/AT_"$re".0.txt &
    GPU_index=$(((GPU_index+1)%num_GPUs))
  done
elif [ "$exp_name" == "schrodinger" ]; then
  ./run_sensitivity.sh &
  exit 0
else
  CUDA_VISIBLE_DEVICES=${GPUs[GPU_index]} DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py \
  &> experiments/"$exp_name"/PINN.txt &
  GPU_index=$(((GPU_index+1)%num_GPUs))
  CUDA_VISIBLE_DEVICES=${GPUs[GPU_index]} DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py \
  --annealing &> experiments/"$exp_name"/PINN-A.txt &
  GPU_index=$(((GPU_index+1)%num_GPUs))
  CUDA_VISIBLE_DEVICES=${GPUs[GPU_index]} DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py \
  --resample &> experiments/"$exp_name"/LWIS.txt &
  GPU_index=$(((GPU_index+1)%num_GPUs))
  CUDA_VISIBLE_DEVICES=${GPUs[GPU_index]} DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py \
  --adversarial &> experiments/"$exp_name"/AT.txt &
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
./draw.sh "$exp_name" &
set +e
num_jobs=$(jobs | grep -c "")
while [ "$num_jobs" -ge 1 ]
do
  num_jobs=$(jobs | grep -c "Run")
  echo "drawing" "$exp_name"
  sleep 2
done
set -e
echo "bash complete"

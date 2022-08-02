#!/bin/bash
set -e
trap 'trap - SIGTERM && kill -- -$$' SIGINT SIGTERM
exp_name="schrodinger"
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
num_train_samples_domain=3000
num_train_samples_boundary=600
data_multipliers=(1 2 4)
sigmas=(0.05 0.1 0.2)
for data_multiplier in "${data_multipliers[@]}"; do
  current_num_train_samples_domain=$((data_multiplier*num_train_samples_domain))
  current_num_train_samples_boundary=$((data_multiplier*num_train_samples_boundary))
  CUDA_VISIBLE_DEVICES=${GPUs[GPU_index]} DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py \
  --num-train-samples-domain ${current_num_train_samples_domain} \
  --num-train-samples-boundary ${current_num_train_samples_boundary} \
  &> experiments/"$exp_name"/PINN_${current_num_train_samples_domain}.txt &
  GPU_index=$(((GPU_index+1)%num_GPUs))
done
for data_multiplier in "${data_multipliers[@]}"; do
  current_num_train_samples_domain=$((data_multiplier*num_train_samples_domain))
  current_num_train_samples_boundary=$((data_multiplier*num_train_samples_boundary))
  CUDA_VISIBLE_DEVICES=${GPUs[GPU_index]} DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py \
  --num-train-samples-domain ${current_num_train_samples_domain} \
  --num-train-samples-boundary ${current_num_train_samples_boundary} \
  --adversarial &> experiments/"$exp_name"/AT_${current_num_train_samples_domain}.txt &
  GPU_index=$(((GPU_index+1)%num_GPUs))
done
set +e
num_jobs=$(jobs | grep -c "")
while [ "$num_jobs" -ge 1 ]
do
  num_jobs=$(jobs | grep -c "Run")
  echo "$exp_name" "$num_jobs" "jobs remaining"
  sleep 2
done
set -e
for data_multiplier in "${data_multipliers[@]}"; do
  current_num_train_samples_domain=$((data_multiplier*num_train_samples_domain))
  current_num_train_samples_boundary=$((data_multiplier*num_train_samples_boundary))
  for sigma in "${sigmas[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPUs[GPU_index]} DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py \
    --num-train-samples-domain ${current_num_train_samples_domain} \
    --num-train-samples-boundary ${current_num_train_samples_boundary} \
    --sigma "${sigma}" --resample &> experiments/"$exp_name"/LWIS_"${current_num_train_samples_domain}"_"${sigma}".txt &
    GPU_index=$(((GPU_index+1)%num_GPUs))
  done
done
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

#!/bin/bash
set -e
exp_name="schrodinger"
GPUs=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
sort -nk 2 -r | awk '$2>3000 {print $1}' | tr -d "\n")
IFS="," read -r -a GPUs <<< "$GPUs"
num_GPUs=${#GPUs[@]}
GPU_index=0
if [ $num_GPUs -eq 0 ]; then
  echo "No enough GPU memory"
  exit 0
fi
bash clean.sh "$exp_name"
num_train_samples_domain=10000
resample_times=5
resample_numbers=1000
data_multipliers=(1 2 4)
sigmas=(0.05 0.1 0.2)
for data_multiplier in "${data_multipliers[@]}"; do
  num_train_samples=$((data_multiplier*(num_train_samples_domain+resample_times*resample_numbers)))
  CUDA_VISIBLE_DEVICES=${GPUs[GPU_index]} DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py \
  --num-train-samples-domain ${num_train_samples} --resample-numbers 0 &> \
  experiments/"$exp_name"/PINN_${num_train_samples}.txt &
  GPU_index=$(((GPU_index+1)%num_GPUs))
done
for data_multiplier in "${data_multipliers[@]}"; do
  num_train_samples=$((data_multiplier*(num_train_samples_domain+resample_times*resample_numbers)))
  for sigma in "${sigmas[@]}"; do
    current_num_train_samples_domain=$((data_multiplier*num_train_samples_domain))
    current_resample_numbers=$((data_multiplier*resample_numbers))
    CUDA_VISIBLE_DEVICES=${GPUs[GPU_index]} DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py \
    --num-train-samples-domain "${current_num_train_samples_domain}" --resample-numbers "${current_resample_numbers}" \
    --resample-times "${resample_times}" --sigma "${sigma}" --resample &> \
    experiments/"$exp_name"/LWIS_"${num_train_samples}"_"${sigma}".txt &
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
echo "bash complete"

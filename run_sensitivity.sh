#!/bin/bash
set -e
exp_name="schrodinger"
bash clean.sh "$exp_name"
num_train_samples_domain=10000
resample_times=5
resample_numbers=1000
data_multipliers=(1 2 4)
sigmas=(0.05 0.10 0.20)
GPU_index=0
for i in $(seq ${#data_multipliers[@]}); do
  num_train_samples=$((data_multipliers[i] * (num_train_samples_domain+resample_times*resample_numbers)))
  CUDA_VISIBLE_DEVICES=$GPU_index DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py \
  --num-train-samples-domain ${num_train_samples} --resample-numbers 0 &> \
  experiments/"$exp_name"/PINN_${num_train_samples}.txt &
  GPU_index=$(((GPU_index+1) % 8))
done
for i in $(seq ${#data_multipliers[@]}); do
  num_train_samples=$((data_multipliers[i] * (num_train_samples_domain+resample_times*resample_numbers)))
  for j in $(seq ${#sigmas[@]}); do
    current_sigma=${sigmas[j]}
    current_num_train_samples_domain=$((data_multipliers[i] * num_train_samples_domain))
    current_resample_numbers=$((data_multipliers[i] * resample_numbers))
    CUDA_VISIBLE_DEVICES=$GPU_index DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py \
    --num-train-samples-domain "${current_num_train_samples_domain}" --resample-numbers "${current_resample_numbers}" \
    --resample-times "${resample_times}" --sigma "${current_sigma}" --resample &> \
    experiments/"$exp_name"/LWIS_"${num_train_samples}"_"${sigma}".txt &
    GPU_index=$(((GPU_index+1) % 8))
  done
done
num_jobs=$(jobs | grep -c "")
while [ "$num_jobs" -ge 1 ]
do
  num_jobs=$(jobs | grep -c "Run")
  echo "$exp_name" "$num_jobs" "jobs remaining"
  sleep 2
done
draw_load=()
for i in $(seq ${#data_multipliers[@]}); do
  num_train_samples=$((data_multipliers[i] * (num_train_samples_domain+resample_times*resample_numbers)))
  draw_load=("${draw_load[@]}" "PINN_${num_train_samples}")
done
for i in $(seq ${#data_multipliers[@]}); do
  num_train_samples=$((data_multipliers[i] * (num_train_samples_domain+resample_times*resample_numbers)))
  for j in $(seq ${#sigmas[@]}); do
    sigma=${sigmas[j]}
    draw_load=("${draw_load[@]}" "LWIS_${num_train_samples}_${sigma}")
  done
done
CUDA_VISIBLE_DEVICES=$GPU_index DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py \
--load "${draw_load[@]}" &> experiments/"$exp_name"/draw_sensitivity.txt &
echo "bash complete"

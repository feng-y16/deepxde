#!/bin/bash
exp_name="schrodinger"
bash clean.sh "$exp_name"
num_train_samples_domain=5000
resample_times=(1 2 3 4)
resample_numbers=5000
sigmas=(0.1 0.5 1.0)
GPU_index=0
for i in $(seq 0 3); do
  num_train_samples=$((num_train_samples_domain+resample_times[i]*resample_numbers))
  CUDA_VISIBLE_DEVICES=$GPU_index DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py \
  --num-train-samples-domain ${num_train_samples} &> experiments/"$exp_name"/PINN_${num_train_samples}.txt &
  GPU_index=$(((GPU_index+1) % 8))
done
for i in $(seq 0 3); do
  num_train_samples=$((num_train_samples_domain+resample_times[i]*resample_numbers))
  for j in $(seq 0 2); do
    resample_time=${resample_times[i]}
    sigma=${sigmas[j]}
    CUDA_VISIBLE_DEVICES=$GPU_index DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py \
    --num-train-samples-domain "${num_train_samples_domain}" --resample-numbers "${resample_numbers}" \
    --resample_times "${resample_time}" --sigma "${sigma}" --resample &> \
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
draw_load=""
for i in $(seq 0 3); do
  num_train_samples=$((num_train_samples_domain+resample_times[i]*resample_numbers))
  draw_load="${draw_load} PINN_${num_train_samples}"
done
for i in $(seq 0 3); do
  num_train_samples=$((num_train_samples_domain+resample_times[i]*resample_numbers))
  for j in $(seq 0 2); do
    sigma=${sigmas[j]}
    draw_load="${draw_load} LWIS_${num_train_samples}_${sigma}"
  done
done
draw_load="${draw_load} "
draw_load=${draw_load:1:-1}
CUDA_VISIBLE_DEVICES=$GPU_index DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py \
--load "${draw_load}" &> experiments/"$exp_name"/draw_sensitivity.txt &
echo "bash complete"

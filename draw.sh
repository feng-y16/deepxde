#!/bin/bash
exp_name=$1
if [ "$exp_name" == "navier_stokes" ]; then
  CUDA_VISIBLE_DEVICES=0 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py \
  --load PINN_100.0 LWIS_100.0 --re 100 &> experiments/"$exp_name"/draw_100.txt &
  CUDA_VISIBLE_DEVICES=1 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py \
  --load PINN_1000.0 LWIS_1000.0 --re 1000 &> experiments/"$exp_name"/draw_1000.txt &
  CUDA_VISIBLE_DEVICES=2 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py \
  --load PINN_10000.0 LWIS_10000.0 --re 10000 &> experiments/"$exp_name"/draw_10000.txt &
elif [ "$exp_name" == "schrodinger" ]; then
  num_train_samples_domain=2000
  resample_times=(4 9 14 19)
  resample_numbers=2000
  sigmas=(0.05 0.1 0.2)
  draw_load=()
  for i in $(seq 0 3); do
    num_train_samples=$((num_train_samples_domain+resample_times[i]*resample_numbers))
    draw_load=("${draw_load[@]}" "PINN_${num_train_samples}")
  done
  for i in $(seq 0 3); do
    num_train_samples=$((num_train_samples_domain+resample_times[i]*resample_numbers))
    for j in $(seq 0 2); do
      sigma=${sigmas[j]}
      draw_load=("${draw_load[@]}" "LWIS_${num_train_samples}_${sigma}")
    done
  done
  CUDA_VISIBLE_DEVICES=3 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py \
  --load "${draw_load[@]}" &> experiments/"$exp_name"/draw_sensitivity.txt &
else
  CUDA_VISIBLE_DEVICES=4 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --load PINN LWIS \
  &> experiments/"$exp_name"/draw.txt &
fi

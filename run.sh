#!/bin/bash
exp_name=$1
bash clean.sh "$exp_name"
if [ "$exp_name" == "navier_stokes" ]; then
  CUDA_VISIBLE_DEVICES=0 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --re 100 \
  &> experiments/"$exp_name"/PINN_100.txt &
  CUDA_VISIBLE_DEVICES=1 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --resample --re 100 \
  &> experiments/"$exp_name"/LWIS_100.txt &
  CUDA_VISIBLE_DEVICES=2 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --re 1000 \
  &> experiments/"$exp_name"/PINN_1000.txt &
  CUDA_VISIBLE_DEVICES=3 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --resample --re 1000 \
  &> experiments/"$exp_name"/LWIS_1000.txt &
  CUDA_VISIBLE_DEVICES=4 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --re 10000 \
  &> experiments/"$exp_name"/PINN_10000.txt &
  CUDA_VISIBLE_DEVICES=5 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --resample --re 10000 \
  &> experiments/"$exp_name"/LWIS_10000.txt &
elif [ "$exp_name" == "schrodinger" ]; then
  bash run_sensitivity.sh
  exit 0
else
  CUDA_VISIBLE_DEVICES=6 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py \
  &> experiments/"$exp_name"/PINN.txt &
  CUDA_VISIBLE_DEVICES=7 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py \
  --resample &> experiments/"$exp_name"/LWIS.txt &
fi
num_jobs=$(jobs | grep -c "")
while [ "$num_jobs" -ge 1 ]
do
  num_jobs=$(jobs | grep -c "Run")
  echo "$exp_name" "$num_jobs" "jobs remaining"
  sleep 2
done
bash draw.sh "$exp_name"
echo "$exp_name" "complete"

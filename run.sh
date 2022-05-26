#!/bin/bash
exp_name=$1
bash clean.sh "$exp_name"
if [ "$exp_name" == "navier_stokes" ]; then
  CUDA_VISIBLE_DEVICES=0 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --re 100 &> experiments/"$exp_name"/PINN_100.txt &
  CUDA_VISIBLE_DEVICES=1 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --resample --re 100 &> experiments/"$exp_name"/LWIS_100.txt &
  CUDA_VISIBLE_DEVICES=2 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --re 200 &> experiments/"$exp_name"/PINN_200.txt &
  CUDA_VISIBLE_DEVICES=3 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --resample --re 200 &> experiments/"$exp_name"/LWIS_200.txt &
  CUDA_VISIBLE_DEVICES=4 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --re 500 &> experiments/"$exp_name"/PINN_500.txt &
  CUDA_VISIBLE_DEVICES=5 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --resample --re 500 &> experiments/"$exp_name"/LWIS_500.txt &
else
  CUDA_VISIBLE_DEVICES=6 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py &> experiments/"$exp_name"/PINN.txt &
  CUDA_VISIBLE_DEVICES=7 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --resample &> experiments/"$exp_name"/LWIS.txt &
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

#!/bin/bash
exp_name=$1
rm -rf experiments/"$exp_name"/*.pkl
rm -rf experiments/"$exp_name"/*.png
rm -rf experiments/"$exp_name"/*.pdf
rm -rf experiments/"$exp_name"/*.txt
rm -rf experiments/"$exp_name"/*.log
CUDA_VISIBLE_DEVICES=4 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py >> experiments/"$exp_name"/PINN.txt &
CUDA_VISIBLE_DEVICES=5 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --resample >> experiments/"$exp_name"/LWIS.txt &
num_jobs=$(jobs | grep -c "")
while [ "$num_jobs" -ge 1 ]
do
  jobs
  num_jobs=$(jobs | grep -c "")
  sleep 10
done
CUDA_VISIBLE_DEVICES=4 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --load PINN LWIS >> experiments/"$exp_name"/draw.txt &

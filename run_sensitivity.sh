#!/bin/bash
exp_name="schrodinger"
bash clean.sh "$exp_name"
CUDA_VISIBLE_DEVICES=0 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --num-train-samples-domain 15000 --resample-numbers 0 > experiments/"$exp_name"/PINN_15000.txt &
CUDA_VISIBLE_DEVICES=1 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --num-train-samples-domain 30000 --resample-numbers 0 > experiments/"$exp_name"/PINN_30000.txt &
CUDA_VISIBLE_DEVICES=2 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --num-train-samples-domain 45000 --resample-numbers 0 > experiments/"$exp_name"/PINN_45000.txt &
CUDA_VISIBLE_DEVICES=3 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --num-train-samples-domain 60000 --resample-numbers 0 > experiments/"$exp_name"/PINN_60000.txt &
CUDA_VISIBLE_DEVICES=4 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --num-train-samples-domain 10000 --resample-numbers 500 --resample --sigma 0.5 > experiments/"$exp_name"/LWIS_15000_0.5.txt &
CUDA_VISIBLE_DEVICES=5 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --num-train-samples-domain 20000 --resample-numbers 1000 --resample --sigma 0.5 > experiments/"$exp_name"/LWIS_30000_0.5.txt &
CUDA_VISIBLE_DEVICES=6 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --num-train-samples-domain 30000 --resample-numbers 1500 --resample --sigma 0.5 > experiments/"$exp_name"/LWIS_45000_0.5.txt &
CUDA_VISIBLE_DEVICES=7 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --num-train-samples-domain 40000 --resample-numbers 2000 --resample --sigma 0.5 > experiments/"$exp_name"/LWIS_60000_0.5.txt &
CUDA_VISIBLE_DEVICES=0 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --num-train-samples-domain 10000 --resample-numbers 500 --resample --sigma 1.0 > experiments/"$exp_name"/LWIS_15000_1.0.txt &
CUDA_VISIBLE_DEVICES=1 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --num-train-samples-domain 20000 --resample-numbers 1000 --resample --sigma 1.0 > experiments/"$exp_name"/LWIS_30000_1.0.txt &
CUDA_VISIBLE_DEVICES=2 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --num-train-samples-domain 30000 --resample-numbers 1500 --resample --sigma 1.0 > experiments/"$exp_name"/LWIS_45000_1.0.txt &
CUDA_VISIBLE_DEVICES=3 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --num-train-samples-domain 40000 --resample-numbers 2000 --resample --sigma 1.0 > experiments/"$exp_name"/LWIS_60000_1.0.txt &
CUDA_VISIBLE_DEVICES=4 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --num-train-samples-domain 10000 --resample-numbers 500 --resample --sigma 2.0 > experiments/"$exp_name"/LWIS_15000_2.0.txt &
CUDA_VISIBLE_DEVICES=5 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --num-train-samples-domain 20000 --resample-numbers 1000 --resample --sigma 2.0 > experiments/"$exp_name"/LWIS_30000_2.0.txt &
CUDA_VISIBLE_DEVICES=6 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --num-train-samples-domain 30000 --resample-numbers 1500 --resample --sigma 2.0 > experiments/"$exp_name"/LWIS_45000_2.0.txt &
CUDA_VISIBLE_DEVICES=7 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --num-train-samples-domain 40000 --resample-numbers 2000 --resample --sigma 2.0 > experiments/"$exp_name"/LWIS_60000_2.0.txt &

num_jobs=$(jobs | grep -c "")
while [ "$num_jobs" -ge 1 ]
do
  jobs
  num_jobs=$(jobs | grep -c "")
  sleep 10
done
CUDA_VISIBLE_DEVICES=6 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --load PINN_15000 PINN_30000 PINN_45000 PINN_60000 \
  LWIS_15000_0.5 LWIS_15000_1.0 LWIS_15000_2.0 LWIS_30000_0.5 LWIS_30000_1.0 LWIS_30000_2.0 LWIS_45000_0.5 LWIS_45000_1.0 LWIS_45000_2.0 \
  LWIS_60000_0.5 LWIS_60000_1.0 LWIS_60000_2.0 > experiments/"$exp_name"/draw_sensitivity.txt &
echo "bash complete"

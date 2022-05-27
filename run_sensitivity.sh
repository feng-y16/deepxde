#!/bin/bash
exp_name="schrodinger"
bash clean.sh "$exp_name"
CUDA_VISIBLE_DEVICES=0 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --num-train-samples-domain 20000 --resample-numbers 0 &> experiments/"$exp_name"/PINN_20000.txt &
CUDA_VISIBLE_DEVICES=1 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --num-train-samples-domain 30000 --resample-numbers 0 &> experiments/"$exp_name"/PINN_30000.txt &
CUDA_VISIBLE_DEVICES=2 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --num-train-samples-domain 40000 --resample-numbers 0 &> experiments/"$exp_name"/PINN_40000.txt &
CUDA_VISIBLE_DEVICES=3 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --num-train-samples-domain 50000 --resample-numbers 0 &> experiments/"$exp_name"/PINN_50000.txt &
CUDA_VISIBLE_DEVICES=4 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --num-train-samples-domain 10000 --resample-times 1 --resample --sigma 0.1 &> experiments/"$exp_name"/LWIS_20000_0.1.txt &
CUDA_VISIBLE_DEVICES=5 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --num-train-samples-domain 10000 --resample-times 2 --resample --sigma 0.1 &> experiments/"$exp_name"/LWIS_30000_0.1.txt &
CUDA_VISIBLE_DEVICES=6 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --num-train-samples-domain 10000 --resample-times 3 --resample --sigma 0.1 &> experiments/"$exp_name"/LWIS_40000_0.1.txt &
CUDA_VISIBLE_DEVICES=7 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --num-train-samples-domain 10000 --resample-times 4 --resample --sigma 0.1 &> experiments/"$exp_name"/LWIS_50000_0.1.txt &
CUDA_VISIBLE_DEVICES=0 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --num-train-samples-domain 10000 --resample-times 1 --resample --sigma 0.5 &> experiments/"$exp_name"/LWIS_20000_0.5.txt &
CUDA_VISIBLE_DEVICES=1 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --num-train-samples-domain 10000 --resample-times 2 --resample --sigma 0.5 &> experiments/"$exp_name"/LWIS_30000_0.5.txt &
CUDA_VISIBLE_DEVICES=2 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --num-train-samples-domain 10000 --resample-times 3 --resample --sigma 0.5 &> experiments/"$exp_name"/LWIS_40000_0.5.txt &
CUDA_VISIBLE_DEVICES=3 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --num-train-samples-domain 10000 --resample-times 4 --resample --sigma 0.5 &> experiments/"$exp_name"/LWIS_50000_0.5.txt &
CUDA_VISIBLE_DEVICES=4 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --num-train-samples-domain 10000 --resample-times 1 --resample --sigma 1.0 &> experiments/"$exp_name"/LWIS_20000_1.0.txt &
CUDA_VISIBLE_DEVICES=5 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --num-train-samples-domain 10000 --resample-times 2 --resample --sigma 1.0 &> experiments/"$exp_name"/LWIS_30000_1.0.txt &
CUDA_VISIBLE_DEVICES=6 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --num-train-samples-domain 10000 --resample-times 3 --resample --sigma 1.0 &> experiments/"$exp_name"/LWIS_40000_1.0.txt &
CUDA_VISIBLE_DEVICES=7 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --num-train-samples-domain 10000 --resample-times 4 --resample --sigma 1.0 &> experiments/"$exp_name"/LWIS_50000_1.0.txt &

num_jobs=$(jobs | grep -c "")
while [ "$num_jobs" -ge 1 ]
do
  jobs
  num_jobs=$(jobs | grep -c "")
  sleep 1
done
CUDA_VISIBLE_DEVICES=6 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --load PINN_20000 PINN_30000 PINN_40000 PINN_50000 \
  LWIS_20000_0.1 LWIS_20000_0.5 LWIS_20000_1.0 LWIS_30000_0.1 LWIS_30000_0.5 LWIS_30000_1.0 LWIS_40000_0.1 LWIS_40000_0.5 LWIS_40000_1.0 \
  LWIS_50000_0.1 LWIS_50000_0.5 LWIS_50000_1.0 &> experiments/"$exp_name"/draw_sensitivity.txt &
echo "bash complete"

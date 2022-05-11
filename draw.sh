#!/bin/bash
exp_name=$1
if [ "$exp_name" == "navier_stokes" ]; then
  CUDA_VISIBLE_DEVICES=0 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --load PINN_100.0 LWIS_100.0 --re 100 > experiments/"$exp_name"/draw_100.txt &
  CUDA_VISIBLE_DEVICES=2 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --load PINN_200.0 LWIS_200.0 --re 200 > experiments/"$exp_name"/draw_200.txt &
  CUDA_VISIBLE_DEVICES=4 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --load PINN_500.0 LWIS_500.0 --re 500 > experiments/"$exp_name"/draw_500.txt &
else
  CUDA_VISIBLE_DEVICES=6 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --load PINN LWIS > experiments/"$exp_name"/draw.txt &
fi

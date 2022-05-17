#!/bin/bash
exp_name=$1
if [ "$exp_name" == "navier_stokes" ]; then
  CUDA_VISIBLE_DEVICES=0 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --load PINN_100.0 LWIS_100.0 --re 100 > experiments/"$exp_name"/draw_100.txt &
  CUDA_VISIBLE_DEVICES=1 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --load PINN_200.0 LWIS_200.0 --re 200 > experiments/"$exp_name"/draw_200.txt &
  CUDA_VISIBLE_DEVICES=2 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --load PINN_500.0 LWIS_500.0 --re 500 > experiments/"$exp_name"/draw_500.txt &
elif [ "$exp_name" == "schrodinger" ]; then
  CUDA_VISIBLE_DEVICES=3 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --load PINN_30000 LWIS_30000_1.0 > experiments/"$exp_name"/draw.txt &
  CUDA_VISIBLE_DEVICES=4 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --load PINN_15000 PINN_30000 PINN_45000 PINN_60000 \
  LWIS_15000_0.5 LWIS_15000_1.0 LWIS_15000_2.0 LWIS_30000_0.5 LWIS_30000_1.0 LWIS_30000_2.0 LWIS_45000_0.5 LWIS_45000_1.0 LWIS_45000_2.0 \
  LWIS_60000_0.5 LWIS_60000_1.0 LWIS_60000_2.0 > experiments/"$exp_name"/draw_sensitivity.txt &
else
  CUDA_VISIBLE_DEVICES=5 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --load PINN LWIS > experiments/"$exp_name"/draw.txt &
fi

#!/bin/bash
set -e
trap 'trap - SIGTERM && kill -- -$$' SIGINT SIGTERM
exp_name=navier_stokes
device=8
re=10
./clean.sh $exp_name
CUDA_VISIBLE_DEVICES=$device DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py \
  --resample --resample-ratio 0.1 --resample-every 1 --re $re --epochs 20000
CUDA_VISIBLE_DEVICES=$device DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py \
  --resample --load LWIS_$re.0 --re $re

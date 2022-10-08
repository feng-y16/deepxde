#!/bin/bash
set -e
trap 'trap - SIGTERM && kill -- -$$' SIGINT SIGTERM
exp_name=burgers
./clean.sh $exp_name
CUDA_VISIBLE_DEVICES=0 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --resample --domain-only
CUDA_VISIBLE_DEVICES=0 DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py --load LWIS-D
echo "debug complete"

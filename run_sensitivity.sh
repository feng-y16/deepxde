#!/bin/bash
set -e
trap 'trap - SIGTERM && kill -- -$$' SIGINT SIGTERM
exp_name=$1
GPUs=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
sort -nk 2 -r | awk '$2>3000 {print $1}' | tr -d "\n")
IFS="," read -r -a GPUs <<< "$GPUs"
num_GPUs=${#GPUs[@]}
GPU_index=0
if [ "$num_GPUs" -eq 0 ]; then
  echo "No enough GPU memory"
  exit 0
fi
resample_splits=(1 2 4)
sigmas=(0.01 0.1 1.0)
for resample_split in "${resample_splits[@]}"; do
  for sigma in "${sigmas[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPUs[GPU_index]} DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py \
    --sigma "${sigma}" --resample --resample-splits "${resample_split}" --sensitivity \
    &> experiments/"$exp_name"/LWIS_"${resample_split}"_"${sigma}".txt &
    GPU_index=$(((GPU_index+1)%num_GPUs))
  done
done
set +e
num_jobs=$(jobs | grep -c "")
while [ "$num_jobs" -ge 1 ]
do
  num_jobs=$(jobs | grep -c "Run")
  echo "$exp_name" "$num_jobs" "jobs remaining"
  sleep 2
done
set -e
./draw_sensitivity.sh "$exp_name" &
set +e
num_jobs=$(jobs | grep -c "")
while [ "$num_jobs" -ge 1 ]
do
  num_jobs=$(jobs | grep -c "Run")
  echo "drawing" "$exp_name"
  sleep 2
done
set -e
echo "bash complete"

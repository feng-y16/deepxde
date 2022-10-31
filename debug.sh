exp_name=burgers
device=7
#./clean.sh $exp_name
#CUDA_VISIBLE_DEVICES=$device DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py \
#  --resample --domain-only --resample-ratio 0.0 --resample-times 2000 --epochs 20000
CUDA_VISIBLE_DEVICES=$device DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py \
  --resample --domain-only --load LWIS-D

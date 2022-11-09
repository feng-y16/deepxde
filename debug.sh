exp_name=burgers
device=8
./clean.sh $exp_name
CUDA_VISIBLE_DEVICES=$device DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py \
  --resample --resample-ratio 0.5 --resample-every 5 --epochs 20000
CUDA_VISIBLE_DEVICES=$device DDEBACKEND=tensorflow python experiments/"$exp_name"/"$exp_name".py \
  --resample --load LWIS

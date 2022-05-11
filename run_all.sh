#!/bin/bash
bash clean_all.sh
CUDA_VISIBLE_DEVICES=0 DDEBACKEND=tensorflow python experiments/ode_system/ode_system.py > experiments/ode_system/PINN.txt &
CUDA_VISIBLE_DEVICES=1 DDEBACKEND=tensorflow python experiments/ode_system/ode_system.py --resample > experiments/ode_system/LWIS.txt &
CUDA_VISIBLE_DEVICES=2 DDEBACKEND=tensorflow python experiments/burgers/burgers.py > experiments/burgers/PINN.txt &
CUDA_VISIBLE_DEVICES=3 DDEBACKEND=tensorflow python experiments/burgers/burgers.py --resample > experiments/burgers/LWIS.txt &
CUDA_VISIBLE_DEVICES=4 DDEBACKEND=tensorflow python experiments/schrodinger/schrodinger.py > experiments/schrodinger/PINN.txt &
CUDA_VISIBLE_DEVICES=5 DDEBACKEND=tensorflow python experiments/schrodinger/schrodinger.py --resample > experiments/schrodinger/LWIS.txt &
CUDA_VISIBLE_DEVICES=6 DDEBACKEND=tensorflow python experiments/navier_stokes/navier_stokes.py --re 100 > experiments/navier_stokes/PINN_100.txt &
CUDA_VISIBLE_DEVICES=7 DDEBACKEND=tensorflow python experiments/navier_stokes/navier_stokes.py --resample --re 100 -- > experiments/navier_stokes/LWIS_100.txt &
CUDA_VISIBLE_DEVICES=0 DDEBACKEND=tensorflow python experiments/navier_stokes/navier_stokes.py --re 200 > experiments/navier_stokes/PINN_200.txt &
CUDA_VISIBLE_DEVICES=1 DDEBACKEND=tensorflow python experiments/navier_stokes/navier_stokes.py --resample --re 200 > experiments/navier_stokes/LWIS_200.txt &
CUDA_VISIBLE_DEVICES=2 DDEBACKEND=tensorflow python experiments/navier_stokes/navier_stokes.py --re 500 > experiments/navier_stokes/PINN_500.txt &
CUDA_VISIBLE_DEVICES=3 DDEBACKEND=tensorflow python experiments/navier_stokes/navier_stokes.py --resample --re 500 > experiments/navier_stokes/LWIS_500.txt &
num_jobs=$(jobs | grep -c "")
while [ "$num_jobs" -ge 1 ]
do
  jobs
  num_jobs=$(jobs | grep -c "")
  sleep 10
done
bash draw_all.sh
echo "bash complete"

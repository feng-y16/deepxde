#!/bin/bash
bash clean.sh
CUDA_VISIBLE_DEVICES=0 DDEBACKEND=tensorflow python experiments/ode_system/ode_system.py >> experiments/ode_system/PINN.log &
CUDA_VISIBLE_DEVICES=1 DDEBACKEND=tensorflow python experiments/ode_system/ode_system.py --resample >> experiments/ode_system/LWIS.log &
# CUDA_VISIBLE_DEVICES=2 DDEBACKEND=tensorflow python experiments/schrodinger/schrodinger.py >> experiments/schrodinger/PINN.log &
# CUDA_VISIBLE_DEVICES=3 DDEBACKEND=tensorflow python experiments/schrodinger/schrodinger.py --resample >> experiments/schrodinger/LWIS.log &
CUDA_VISIBLE_DEVICES=4 DDEBACKEND=tensorflow python experiments/burgers/burgers.py >> experiments/burgers/PINN.log &
CUDA_VISIBLE_DEVICES=5 DDEBACKEND=tensorflow python experiments/burgers/burgers.py --resample >> experiments/burgers/LWIS.log &
CUDA_VISIBLE_DEVICES=6 DDEBACKEND=tensorflow python experiments/navier_stokes/navier_stokes.py >> experiments/navier_stokes/PINN.log &
CUDA_VISIBLE_DEVICES=7 DDEBACKEND=tensorflow python experiments/navier_stokes/navier_stokes.py --resample >> experiments/navier_stokes/LWIS.log &
while [ "$(jobs | grep -c "")" -ge 1 ]
do
sleep 60
done
bash plot.sh
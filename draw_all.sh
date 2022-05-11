#!/bin/bash
CUDA_VISIBLE_DEVICES=0 DDEBACKEND=tensorflow python experiments/ode_system/ode_system.py --load PINN LWIS > experiments/ode_system/draw.txt &
CUDA_VISIBLE_DEVICES=1 DDEBACKEND=tensorflow python experiments/schrodinger/schrodinger.py --load PINN LWIS > experiments/schrodinger/draw.txt &
CUDA_VISIBLE_DEVICES=2 DDEBACKEND=tensorflow python experiments/burgers/burgers.py --load PINN LWIS > experiments/burgers/draw.txt &
CUDA_VISIBLE_DEVICES=3 DDEBACKEND=tensorflow python experiments/navier_stokes/navier_stokes.py --load PINN_100 LWIS_100 > experiments/navier_stokes/draw_100.txt &
CUDA_VISIBLE_DEVICES=4 DDEBACKEND=tensorflow python experiments/navier_stokes/navier_stokes.py --load PINN_200 LWIS_200 > experiments/navier_stokes/draw_200.txt &
CUDA_VISIBLE_DEVICES=5 DDEBACKEND=tensorflow python experiments/navier_stokes/navier_stokes.py --load PINN_500 LWIS_500 > experiments/navier_stokes/draw_500.txt &

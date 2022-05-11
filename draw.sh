#!/bin/bash
CUDA_VISIBLE_DEVICES=0 DDEBACKEND=tensorflow python experiments/ode_system/ode_system.py --load PINN LWIS > experiments/ode_system/draw.txt &
CUDA_VISIBLE_DEVICES=1 DDEBACKEND=tensorflow python experiments/schrodinger/schrodinger.py --load PINN LWIS > experiments/schrodinger/draw.txt &
CUDA_VISIBLE_DEVICES=2 DDEBACKEND=tensorflow python experiments/burgers/burgers.py --load PINN LWIS > experiments/burgers/draw.txt &
CUDA_VISIBLE_DEVICES=3 DDEBACKEND=tensorflow python experiments/navier_stokes/navier_stokes.py --load 100_PINN 100_LWIS > experiments/navier_stokes/draw.txt &
CUDA_VISIBLE_DEVICES=4 DDEBACKEND=tensorflow python experiments/navier_stokes/navier_stokes.py --load 1000_PINN 1000_LWIS > experiments/navier_stokes/1000_draw.txt &
CUDA_VISIBLE_DEVICES=5 DDEBACKEND=tensorflow python experiments/navier_stokes/navier_stokes.py --load 10000_PINN 10000_LWIS > experiments/navier_stokes/10000_draw.txt &

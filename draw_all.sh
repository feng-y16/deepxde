#!/bin/bash
CUDA_VISIBLE_DEVICES=0 DDEBACKEND=tensorflow python experiments/stiff_ode/stiff_ode.py --load PINN LWIS > experiments/stiff_ode/draw.txt &
CUDA_VISIBLE_DEVICES=1 DDEBACKEND=tensorflow python experiments/schrodinger/schrodinger.py --load PINN LWIS > experiments/schrodinger/draw.txt &
CUDA_VISIBLE_DEVICES=2 DDEBACKEND=tensorflow python experiments/burgers/burgers.py --load PINN LWIS > experiments/burgers/draw.txt &
CUDA_VISIBLE_DEVICES=3 DDEBACKEND=tensorflow python experiments/navier_stokes/navier_stokes.py --load PINN_100.0 LWIS_100.0 > experiments/navier_stokes/draw_100.txt &
CUDA_VISIBLE_DEVICES=4 DDEBACKEND=tensorflow python experiments/navier_stokes/navier_stokes.py --load PINN_200.0 LWIS_200.0 > experiments/navier_stokes/draw_200.txt &
CUDA_VISIBLE_DEVICES=5 DDEBACKEND=tensorflow python experiments/navier_stokes/navier_stokes.py --load PINN_500.0 LWIS_500.0 > experiments/navier_stokes/draw_500.txt &

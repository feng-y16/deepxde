#!/bin/bash
CUDA_VISIBLE_DEVICES=0 DDEBACKEND=tensorflow python experiments/stiff_ode/stiff_ode.py --load PINN LWIS > experiments/stiff_ode/draw.txt &
CUDA_VISIBLE_DEVICES=1 DDEBACKEND=tensorflow python experiments/schrodinger/schrodinger.py --load PINN_30000 LWIS_30000_1.0 > experiments/schrodinger/draw.txt &
CUDA_VISIBLE_DEVICES=2 DDEBACKEND=tensorflow python experiments/burgers/burgers.py --load PINN LWIS > experiments/burgers/draw.txt &
CUDA_VISIBLE_DEVICES=3 DDEBACKEND=tensorflow python experiments/navier_stokes/navier_stokes.py --load PINN_100.0 LWIS_100.0 > experiments/navier_stokes/draw_100.txt &
CUDA_VISIBLE_DEVICES=4 DDEBACKEND=tensorflow python experiments/navier_stokes/navier_stokes.py --load PINN_200.0 LWIS_200.0 > experiments/navier_stokes/draw_200.txt &
CUDA_VISIBLE_DEVICES=5 DDEBACKEND=tensorflow python experiments/navier_stokes/navier_stokes.py --load PINN_500.0 LWIS_500.0 > experiments/navier_stokes/draw_500.txt &
CUDA_VISIBLE_DEVICES=6 DDEBACKEND=tensorflow python experiments/schrodinger/schrodinger.py --load PINN_15000 PINN_30000 PINN_45000 PINN_60000 \
  LWIS_15000_0.5 LWIS_15000_1.0 LWIS_15000_2.0 LWIS_30000_0.5 LWIS_30000_1.0 LWIS_30000_2.0 LWIS_45000_0.5 LWIS_45000_1.0 LWIS_45000_2.0 \
  LWIS_60000_0.5 LWIS_60000_1.0 LWIS_60000_2.0 > experiments/schrodinger/draw_sensitivity.txt &
CUDA_VISIBLE_DEVICES=7 DDEBACKEND=tensorflow python experiments/convection/convection.py --load PINN LWIS > experiments/convection/draw.txt &

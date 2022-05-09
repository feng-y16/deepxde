CUDA_VISIBLE_DEVICES=0 DDEBACKEND=tensorflow python experiments/ode_system/ode_system.py --load PINN LWIS >> experiments/ode_system/draw.log &
# CUDA_VISIBLE_DEVICES=2 DDEBACKEND=tensorflow python experiments/schrodinger/schrodinger.py --load PINN LWIS >> experiments/schrodinger/draw.log &
CUDA_VISIBLE_DEVICES=4 DDEBACKEND=tensorflow python experiments/burgers/burgers.py --load PINN LWIS >> experiments/burgers/draw.log &
CUDA_VISIBLE_DEVICES=6 DDEBACKEND=tensorflow python experiments/navier_stokes/navier_stokes.py --load PINN LWIS >> experiments/navier_stokes/draw.log &

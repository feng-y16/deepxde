CUDA_VISIBLE_DEVICES=0 DDEBACKEND=tensorflow python experiments/ode_system/ode_system.py
CUDA_VISIBLE_DEVICES=1 DDEBACKEND=tensorflow python experiments/ode_system/ode_system.py --resample
CUDA_VISIBLE_DEVICES=2 DDEBACKEND=tensorflow python experiments/schrodinger/schrodinger.py
CUDA_VISIBLE_DEVICES=3 DDEBACKEND=tensorflow python experiments/schrodinger/schrodinger.py --resample
CUDA_VISIBLE_DEVICES=4 DDEBACKEND=tensorflow python experiments/burgers/Burgers.py
CUDA_VISIBLE_DEVICES=5 DDEBACKEND=tensorflow python experiments/burgers/Burgers.py --resample
CUDA_VISIBLE_DEVICES=6 DDEBACKEND=tensorflow python experiments/navier_stokes/navier_stokes.py
CUDA_VISIBLE_DEVICES=7 DDEBACKEND=tensorflow python experiments/navier_stokes/navier_stokes.py --resample

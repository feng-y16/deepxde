"""
This file was built to solve numerically 1D Burgers' equation wave equation with the FFT. The equation corresponds to :
$\dfrac{\partial u}{\partial t} + \mu u\dfrac{\partial u}{\partial x} = \nu \dfrac{\partial^2 u}{\partial x^2}$

where
 - u represent the signal
 - x represent the position
 - t represent the time
 - nu and mu are constants to balance the non-linear and diffusion terms.
Copyright - © SACHA BINDER - 2021
"""
import pdb
import numpy as np
from burgers import u_func
import os
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def burg_system(u, t, k, mu, nu):
    # Spatial derivative in the Fourier domain
    u_hat = np.fft.fft(u)
    u_hat_x = 1j * k * u_hat
    u_hat_xx = -k ** 2 * u_hat

    # Switching in the spatial domain
    u_x = np.fft.ifft(u_hat_x)
    u_xx = np.fft.ifft(u_hat_xx)

    # ODE resolution
    u_t = -mu * u * u_x + nu * u_xx
    return u_t.real


# def solve(n_points=100, device=torch.device("cuda")):
#     x = torch.linspace(-1.0, 1.0, n_points, device=device)
#     t = torch.linspace(0.0, 1.0, n_points, device=device)
#     Y, X = torch.meshgrid(x, t, indexing='ij')
#     y = [torch.tensor(u_func(x.cpu().numpy().reshape(1, -1), device=device))]
#     for _ in range(n_points):
#         new_y = torch.zeros_like(y)
#         y.append(None)
#     return X, y, t, x


def gen_testdata():
    burgers_data = np.load(os.path.join(save_dir, "Burgers.npz"))
    t, x, exact = burgers_data["t"], burgers_data["x"], burgers_data["usol"].T
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = exact.flatten()[:, None]
    return X, y, t, x


if __name__ == "__main__":
    mu = 1
    nu = 0.01  # kinematic viscosity coefficient
    L_x = 2  # Range of the domain according to x [m]
    dx = 0.001  # Infinitesimal distance
    N_x = int(L_x / dx)  # Points number of the spatial mesh
    X = np.linspace(-L_x / 2, L_x / 2, N_x)  # Spatial array
    L_t = 1  # Duration of simulation [s]
    dt = 0.01 # Infinitesimal time
    N_t = int(L_t / dt)  # Points number of the temporal mesh
    T = np.linspace(0, L_t, N_t)  # Temporal array
    k = 2 * np.pi * np.fft.fftfreq(N_x, d=dx)
    u0 = u_func(X.reshape(-1, 1))[:, 0]
    U = odeint(burg_system, u0, T, args=(k, mu, nu,), mxstep=10000).T
    t = T.reshape(1, -1)
    x = X.reshape(1, -1)
    save_dir = os.path.dirname(os.path.abspath(__file__))
    plt.pcolormesh(t * np.ones_like(x.T), np.ones_like(t) * x.T, U, cmap="rainbow")
    plt.savefig(os.path.join(save_dir, "debug.png"))

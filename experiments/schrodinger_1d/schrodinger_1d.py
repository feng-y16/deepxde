import numpy as np
import deepxde as dde
import os
import argparse
import warnings
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ep", "--epochs", type=int, default=5000)
    parser.add_argument("-ntrd", "--num-train-samples-domain", type=int, default=1000)
    parser.add_argument("-rest", "--resample-times", type=int, default=20)
    parser.add_argument("-resn", "--resample-numbers", type=int, default=10)
    parser.add_argument("-r", "--resample", action="store_true", default=False)
    return parser.parse_known_args()[0]


x_lower = -5
x_upper = 5
t_lower = 0
t_upper = np.pi / 2

# Creation of the 2D domain (for plotting and input)
x = np.linspace(x_lower, x_upper, 256)
t = np.linspace(t_lower, t_upper, 201)
X, T = np.meshgrid(x, t)

# The whole domain flattened
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

# Space and time domains/geometry (for the deepxde model)
space_domain = dde.geometry.Interval(x_lower, x_upper)
time_domain = dde.geometry.TimeDomain(t_lower, t_upper)
geomtime = dde.geometry.GeometryXTime(space_domain, time_domain)


# The "physics-informed" part of the loss
def pde(x, y):
    """
    INPUTS:
        x: x[:,0] is x-coordinate
           x[:,1] is t-coordinate
        y: Network output, in this case:
            y[:,0] is u(x,t) the real part
            y[:,1] is v(x,t) the imaginary part
    OUTPUT:
        The pde in standard form i.e. something that must be zero
    """

    u = y[:, 0:1]
    v = y[:, 1:2]

    # In 'jacobian', i is the output component and j is the input component
    u_t = dde.grad.jacobian(y, x, i=0, j=1)
    v_t = dde.grad.jacobian(y, x, i=1, j=1)

    u_x = dde.grad.jacobian(y, x, i=0, j=0)
    v_x = dde.grad.jacobian(y, x, i=1, j=0)

    # In 'hessian', i and j are both input components.
    # The Hessian could be in principle something like d^2y/dxdt, d^2y/d^2x, etc.
    # The output component is selected by "component"
    u_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    v_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)

    f_u = u_t + 0.5 * v_xx + (u ** 2 + v ** 2) * v
    f_v = v_t - 0.5 * u_xx - (u ** 2 + v ** 2) * u

    return [f_u, f_v]


# Boundary and Initial conditions
# Periodic Boundary conditions
bc_u_0 = dde.icbc.PeriodicBC(
    geomtime, 0, lambda _, on_boundary: on_boundary, derivative_order=0, component=0
)
bc_u_1 = dde.icbc.PeriodicBC(
    geomtime, 0, lambda _, on_boundary: on_boundary, derivative_order=1, component=0
)
bc_v_0 = dde.icbc.PeriodicBC(
    geomtime, 0, lambda _, on_boundary: on_boundary, derivative_order=0, component=1
)
bc_v_1 = dde.icbc.PeriodicBC(
    geomtime, 0, lambda _, on_boundary: on_boundary, derivative_order=1, component=1
)


# Initial conditions
def init_cond_u(x):
    # "2 sech(x)"
    return 2 / np.cosh(x[:, 0:1])


def init_cond_v(x):
    return 0


warnings.filterwarnings("ignore")
args = parse_args()
resample = args.resample
resample_times = args.resample_times
resample_num = args.resample_numbers
epochs = args.epochs
num_train_samples_domain = args.num_train_samples_domain
print("resample:", resample)
print("total data points:", num_train_samples_domain + resample_times * resample_num)


ic_u = dde.icbc.IC(geomtime, init_cond_u, lambda _, on_initial: on_initial, component=0)
ic_v = dde.icbc.IC(geomtime, init_cond_v, lambda _, on_initial: on_initial, component=1)

if resample:
    data = dde.data.TimePDE(
        geomtime,
        pde,
        [bc_u_0, bc_u_1, bc_v_0, bc_v_1, ic_u, ic_v],
        num_domain=num_train_samples_domain,
        num_boundary=20,
        num_initial=200,
        train_distribution="pseudo",
    )
else:
    data = dde.data.TimePDE(
        geomtime,
        pde,
        [bc_u_0, bc_u_1, bc_v_0, bc_v_1, ic_u, ic_v],
        num_domain=num_train_samples_domain + resample_times * resample_num,
        num_boundary=20,
        num_initial=200,
        train_distribution="pseudo",
    )

# Network architecture
net = dde.nn.FNN([2] + [100] * 4 + [2], "tanh", "Glorot normal")

model = dde.Model(data, net)

# To employ a GPU accelerated system is highly encouraged.

model.compile("adam", lr=1e-3, loss="MSE")
if resample:
    resampler = dde.callbacks.PDEGradientAccumulativeResampler(period=epochs // (resample_times + 1) + 1,
                                                               sample_num=resample_num)
    losshistory, train_state = model.train(epochs=epochs, callbacks=[resampler])
else:
    model.train(epochs=epochs)


save_dir = os.path.dirname(os.path.abspath(__file__))
if resample:
    prefix = "ag_"
else:
    prefix = ""
# Make prediction
prediction = model.predict(X_star, operator=None)

u = griddata(X_star, prediction[:, 0], (X, T), method="cubic")
v = griddata(X_star, prediction[:, 1], (X, T), method="cubic")

h = np.sqrt(u ** 2 + v ** 2)


# Plot predictions
fig, ax = plt.subplots(3)

ax[0].set_title("Results")
ax[0].set_ylabel("Real part")
ax[0].imshow(
    u.T,
    interpolation="nearest",
    cmap="viridis",
    extent=[t_lower, t_upper, x_lower, x_upper],
    origin="lower",
    aspect="auto",
)
ax[1].set_ylabel("Imaginary part")
ax[1].imshow(
    v.T,
    interpolation="nearest",
    cmap="viridis",
    extent=[t_lower, t_upper, x_lower, x_upper],
    origin="lower",
    aspect="auto",
)
ax[2].set_ylabel("Amplitude")
ax[2].imshow(
    h.T,
    interpolation="nearest",
    cmap="viridis",
    extent=[t_lower, t_upper, x_lower, x_upper],
    origin="lower",
    aspect="auto",
)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, prefix + "error.png"))
plt.savefig(os.path.join(save_dir, prefix + "error.pdf"))
plt.close()

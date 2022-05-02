"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
import deepxde as dde
import numpy as np
import os
import argparse
import warnings
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ep", "--epochs", type=int, default=5000)
    parser.add_argument("-ntrd", "--num-train-samples-domain", type=int, default=8000)
    parser.add_argument("-rest", "--resample-times", type=int, default=100)
    parser.add_argument("-resn", "--resample-numbers", type=int, default=20)
    parser.add_argument("-nte", "--num-test-samples", type=int, default=51)
    parser.add_argument("-r", "--resample", action="store_true", default=False)
    parser.add_argument("-l", "--load", action="store_true", default=False)
    return parser.parse_known_args()[0]


a = 1
d = 1
Re = 1


def pde(x, u):
    u_vel, v_vel, w_vel, p = u[:, 0:1], u[:, 1:2], u[:, 2:3], u[:, 3:4]

    u_vel_x = dde.grad.jacobian(u, x, i=0, j=0)
    u_vel_y = dde.grad.jacobian(u, x, i=0, j=1)
    u_vel_z = dde.grad.jacobian(u, x, i=0, j=2)
    u_vel_t = dde.grad.jacobian(u, x, i=0, j=3)
    u_vel_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)
    u_vel_yy = dde.grad.hessian(u, x, component=0, i=1, j=1)
    u_vel_zz = dde.grad.hessian(u, x, component=0, i=2, j=2)

    v_vel_x = dde.grad.jacobian(u, x, i=1, j=0)
    v_vel_y = dde.grad.jacobian(u, x, i=1, j=1)
    v_vel_z = dde.grad.jacobian(u, x, i=1, j=2)
    v_vel_t = dde.grad.jacobian(u, x, i=1, j=3)
    v_vel_xx = dde.grad.hessian(u, x, component=1, i=0, j=0)
    v_vel_yy = dde.grad.hessian(u, x, component=1, i=1, j=1)
    v_vel_zz = dde.grad.hessian(u, x, component=1, i=2, j=2)

    w_vel_x = dde.grad.jacobian(u, x, i=2, j=0)
    w_vel_y = dde.grad.jacobian(u, x, i=2, j=1)
    w_vel_z = dde.grad.jacobian(u, x, i=2, j=2)
    w_vel_t = dde.grad.jacobian(u, x, i=2, j=3)
    w_vel_xx = dde.grad.hessian(u, x, component=2, i=0, j=0)
    w_vel_yy = dde.grad.hessian(u, x, component=2, i=1, j=1)
    w_vel_zz = dde.grad.hessian(u, x, component=2, i=2, j=2)

    p_x = dde.grad.jacobian(u, x, i=3, j=0)
    p_y = dde.grad.jacobian(u, x, i=3, j=1)
    p_z = dde.grad.jacobian(u, x, i=3, j=2)

    momentum_x = (
        u_vel_t
        + (u_vel * u_vel_x + v_vel * u_vel_y + w_vel * u_vel_z)
        + p_x
        - 1 / Re * (u_vel_xx + u_vel_yy + u_vel_zz)
    )
    momentum_y = (
        v_vel_t
        + (u_vel * v_vel_x + v_vel * v_vel_y + w_vel * v_vel_z)
        + p_y
        - 1 / Re * (v_vel_xx + v_vel_yy + v_vel_zz)
    )
    momentum_z = (
        w_vel_t
        + (u_vel * w_vel_x + v_vel * w_vel_y + w_vel * w_vel_z)
        + p_z
        - 1 / Re * (w_vel_xx + w_vel_yy + w_vel_zz)
    )
    continuity = u_vel_x + v_vel_y + w_vel_z

    return [momentum_x, momentum_y, momentum_z, continuity]


def u_func(x):
    return (
        -a
        * (
            np.exp(a * x[:, 0:1]) * np.sin(a * x[:, 1:2] + d * x[:, 2:3])
            + np.exp(a * x[:, 2:3]) * np.cos(a * x[:, 0:1] + d * x[:, 1:2])
        )
        * np.exp(-(d ** 2) * x[:, 3:4])
    )


def v_func(x):
    return (
        -a
        * (
            np.exp(a * x[:, 1:2]) * np.sin(a * x[:, 2:3] + d * x[:, 0:1])
            + np.exp(a * x[:, 0:1]) * np.cos(a * x[:, 1:2] + d * x[:, 2:3])
        )
        * np.exp(-(d ** 2) * x[:, 3:4])
    )


def w_func(x):
    return (
        -a
        * (
            np.exp(a * x[:, 2:3]) * np.sin(a * x[:, 0:1] + d * x[:, 1:2])
            + np.exp(a * x[:, 1:2]) * np.cos(a * x[:, 2:3] + d * x[:, 0:1])
        )
        * np.exp(-(d ** 2) * x[:, 3:4])
    )


def p_func(x):
    return (
        -0.5
        * a ** 2
        * (
            np.exp(2 * a * x[:, 0:1])
            + np.exp(2 * a * x[:, 1:2])
            + np.exp(2 * a * x[:, 2:3])
            + 2
            * np.sin(a * x[:, 0:1] + d * x[:, 1:2])
            * np.cos(a * x[:, 2:3] + d * x[:, 0:1])
            * np.exp(a * (x[:, 1:2] + x[:, 2:3]))
            + 2
            * np.sin(a * x[:, 1:2] + d * x[:, 2:3])
            * np.cos(a * x[:, 0:1] + d * x[:, 1:2])
            * np.exp(a * (x[:, 2:3] + x[:, 0:1]))
            + 2
            * np.sin(a * x[:, 2:3] + d * x[:, 0:1])
            * np.cos(a * x[:, 1:2] + d * x[:, 2:3])
            * np.exp(a * (x[:, 0:1] + x[:, 1:2]))
        )
        * np.exp(-2 * d ** 2 * x[:, 3:4])
    )


warnings.filterwarnings("ignore")
args = parse_args()
resample = args.resample
resample_times = args.resample_times
resample_num = args.resample_numbers
epochs = args.epochs
num_train_samples_domain = args.num_train_samples_domain
num_test_samples = args.num_test_samples
load = args.load
save_dir = os.path.dirname(os.path.abspath(__file__))
if resample:
    prefix = "ag"
else:
    prefix = "norm"
print("resample:", resample)
print("total data points:", num_train_samples_domain + resample_times * resample_num)

spatial_domain = dde.geometry.Cuboid(xmin=[-1, -1, -1], xmax=[1, 1, 1])
temporal_domain = dde.geometry.TimeDomain(0, 1)
spatio_temporal_domain = dde.geometry.GeometryXTime(spatial_domain, temporal_domain)

boundary_condition_u = dde.icbc.DirichletBC(
    spatio_temporal_domain, u_func, lambda _, on_boundary: on_boundary, component=0
)
boundary_condition_v = dde.icbc.DirichletBC(
    spatio_temporal_domain, v_func, lambda _, on_boundary: on_boundary, component=1
)
boundary_condition_w = dde.icbc.DirichletBC(
    spatio_temporal_domain, w_func, lambda _, on_boundary: on_boundary, component=2
)

initial_condition_u = dde.icbc.IC(
    spatio_temporal_domain, u_func, lambda _, on_initial: on_initial, component=0
)
initial_condition_v = dde.icbc.IC(
    spatio_temporal_domain, v_func, lambda _, on_initial: on_initial, component=1
)
initial_condition_w = dde.icbc.IC(
    spatio_temporal_domain, w_func, lambda _, on_initial: on_initial, component=2
)

if resample:
    data = dde.data.TimePDE(
        spatio_temporal_domain,
        pde,
        [
            boundary_condition_u,
            boundary_condition_v,
            boundary_condition_w,
            initial_condition_u,
            initial_condition_v,
            initial_condition_w,
        ],
        num_domain=num_train_samples_domain,
        num_boundary=1000,
        num_initial=1000,
        num_test=10000,
    )
else:
    data = dde.data.TimePDE(
        spatio_temporal_domain,
        pde,
        [
            boundary_condition_u,
            boundary_condition_v,
            boundary_condition_w,
            initial_condition_u,
            initial_condition_v,
            initial_condition_w,
        ],
        num_domain=num_train_samples_domain + resample_times * resample_num,
        num_boundary=1000,
        num_initial=1000,
        num_test=10000,
    )

if not load:
    net = dde.nn.FNN([4] + 4 * [50] + [4], "tanh", "Glorot normal")

    model = dde.Model(data, net)

    model.compile("adam", lr=1e-3, loss_weights=[1, 1, 1, 1, 100, 100, 100, 100, 100, 100])
    if resample:
        resampler = dde.callbacks.PDEGradientAccumulativeResampler(period=epochs // (resample_times + 1) + 1,
                                                                   sample_num=resample_num, sigma=0.2)
        losshistory, train_state = model.train(epochs=epochs, callbacks=[resampler])
    else:
        losshistory, train_state = model.train(epochs=epochs)
    info = {"net": net, "train_x_all": data.train_x_all, "train_x": data.train_x, "train_x_bc": data.train_x_bc,
            "train_y": data.train_y, "test_x": data.test_x, "test_y": data.test_y,
            "loss_history": losshistory, "train_state": train_state}
    with open(os.path.join(save_dir, prefix + "_info.pkl"), "wb") as f:
        pickle.dump(info, f)
else:
    with open(os.path.join(save_dir, prefix + "_info.pkl"), "rb") as f:
        info = pickle.load(f)
    net = info["net"]
    data.train_x_all = info["train_x_all"]
    data.train_x = info["train_x"]
    data.train_x_bc = info["train_x_bc"]
    data.train_y = info["train_y"]
    data.test_x = info["test_x"]
    data.test_y = info["test_y"]
    losshistory = info["loss_history"]
    train_state = info["train_state"]
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3, loss_weights=[1, 1, 1, 1, 100, 100, 100, 100, 100, 100])

x, y, z = np.meshgrid(
    np.linspace(-1, 1, num_test_samples), np.linspace(-1, 1, num_test_samples), np.linspace(-1, 1, num_test_samples)
)

X = np.vstack((np.ravel(x), np.ravel(y), np.ravel(z))).T

t_0 = np.zeros(num_test_samples ** 3).reshape(num_test_samples ** 3, 1)
t_1 = np.ones(num_test_samples ** 3).reshape(num_test_samples ** 3, 1)

X_0 = np.hstack((X, t_0))
X_1 = np.hstack((X, t_1))

output_0 = model.predict(X_0)
output_1 = model.predict(X_1)

u_pred_0 = output_0[:, 0].reshape(-1)
v_pred_0 = output_0[:, 1].reshape(-1)
w_pred_0 = output_0[:, 2].reshape(-1)
p_pred_0 = output_0[:, 3].reshape(-1)

u_exact_0 = u_func(X_0).reshape(-1)
v_exact_0 = v_func(X_0).reshape(-1)
w_exact_0 = w_func(X_0).reshape(-1)
p_exact_0 = p_func(X_0).reshape(-1)

u_pred_1 = output_1[:, 0].reshape(-1)
v_pred_1 = output_1[:, 1].reshape(-1)
w_pred_1 = output_1[:, 2].reshape(-1)
p_pred_1 = output_1[:, 3].reshape(-1)

u_exact_1 = u_func(X_1).reshape(-1)
v_exact_1 = v_func(X_1).reshape(-1)
w_exact_1 = w_func(X_1).reshape(-1)
p_exact_1 = p_func(X_1).reshape(-1)

f_0 = model.predict(X_0, operator=pde)
f_1 = model.predict(X_1, operator=pde)

l2_difference_u_0 = dde.metrics.l2_relative_error(u_exact_0, u_pred_0)
l2_difference_v_0 = dde.metrics.l2_relative_error(v_exact_0, v_pred_0)
l2_difference_w_0 = dde.metrics.l2_relative_error(w_exact_0, w_pred_0)
l2_difference_p_0 = dde.metrics.l2_relative_error(p_exact_0, p_pred_0)
residual_0 = np.mean(np.absolute(f_0))

l2_difference_u_1 = dde.metrics.l2_relative_error(u_exact_1, u_pred_1)
l2_difference_v_1 = dde.metrics.l2_relative_error(v_exact_1, v_pred_1)
l2_difference_w_1 = dde.metrics.l2_relative_error(w_exact_1, w_pred_1)
l2_difference_p_1 = dde.metrics.l2_relative_error(p_exact_1, p_pred_1)
residual_1 = np.mean(np.absolute(f_1))

print("Accuracy at t = 0:")
print("Mean residual:", residual_0)
print("L2 relative error in u:", l2_difference_u_0)
print("L2 relative error in v:", l2_difference_v_0)
print("L2 relative error in w:", l2_difference_w_0)
print("\n")
print("Accuracy at t = 1:")
print("Mean residual:", residual_1)
print("L2 relative error in u:", l2_difference_u_1)
print("L2 relative error in v:", l2_difference_v_1)
print("L2 relative error in w:", l2_difference_w_1)


def contour(grid, x, y, z, title, levels=50, vmin=0, vmax=1):
    # plot a contour
    plt.subplot(grid)
    plt.contour(x, y, z, colors='k', linewidths=0.2, levels=levels, vmin=vmin, vmax=vmax)
    plt.contourf(x, y, z, cmap='rainbow', levels=levels, vmin=vmin, vmax=vmax)
    plt.title(title)
    m = plt.cm.ScalarMappable(cmap='rainbow', norm=Normalize(vmin=vmin, vmax=vmax))
    m.set_array(z)
    m.set_clim(vmin, vmax)
    cbar = plt.colorbar(m, pad=0.03, aspect=25, format='%.0e')
    cbar.mappable.set_clim(vmin, vmax)


for i in range(3):
    index = np.where(X[:, i] == 0)[0]
    x = X_1[index]
    y = model.predict(x)
    y[:, -1] -= np.mean(y[:, -1])
    y_gt = np.concatenate((u_func(x), v_func(x), w_func(x), p_func(x) - np.mean(p_func(x))), axis=1)
    relative_error = np.abs(y - y_gt) / (np.abs(y_gt) + 1e-6)
    relative_error = np.where(relative_error > 1, 1, relative_error)
    x = x[:, np.delete(np.arange(4), i, axis=0)][:, :-1]
    plt.figure(figsize=(6, 5))
    gs = GridSpec(2, 2)
    contour(gs[0, 0], np.linspace(-1, 1, num_test_samples), np.linspace(-1, 1, num_test_samples),
            relative_error[:, 0].reshape(num_test_samples, num_test_samples), 'u')
    contour(gs[0, 1], np.linspace(-1, 1, num_test_samples), np.linspace(-1, 1, num_test_samples),
            relative_error[:, 1].reshape(num_test_samples, num_test_samples), 'v')
    contour(gs[1, 0], np.linspace(-1, 1, num_test_samples), np.linspace(-1, 1, num_test_samples),
            relative_error[:, 2].reshape(num_test_samples, num_test_samples), 'w')
    contour(gs[1, 1], np.linspace(-1, 1, num_test_samples), np.linspace(-1, 1, num_test_samples),
            relative_error[:, 3].reshape(num_test_samples, num_test_samples), 'p')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, prefix + "_" + str(i) + "_error.png"))
    plt.close()
    plt.figure(figsize=(6, 5))
    gs = GridSpec(2, 2)
    contour(gs[0, 0], np.linspace(-1, 1, num_test_samples), np.linspace(-1, 1, num_test_samples),
            y[:, 0].reshape(num_test_samples, num_test_samples), 'u')
    contour(gs[0, 1], np.linspace(-1, 1, num_test_samples), np.linspace(-1, 1, num_test_samples),
            y[:, 1].reshape(num_test_samples, num_test_samples), 'v')
    contour(gs[1, 0], np.linspace(-1, 1, num_test_samples), np.linspace(-1, 1, num_test_samples),
            y[:, 2].reshape(num_test_samples, num_test_samples), 'w')
    contour(gs[1, 1], np.linspace(-1, 1, num_test_samples), np.linspace(-1, 1, num_test_samples),
            y[:, 3].reshape(num_test_samples, num_test_samples), 'p')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, prefix + "_" + str(i) + "_y.png"))
    plt.close()
    plt.figure(figsize=(6, 5))
    gs = GridSpec(2, 2)
    contour(gs[0, 0], np.linspace(-1, 1, num_test_samples), np.linspace(-1, 1, num_test_samples),
            y_gt[:, 0].reshape(num_test_samples, num_test_samples), 'u')
    contour(gs[0, 1], np.linspace(-1, 1, num_test_samples), np.linspace(-1, 1, num_test_samples),
            y_gt[:, 1].reshape(num_test_samples, num_test_samples), 'v')
    contour(gs[1, 0], np.linspace(-1, 1, num_test_samples), np.linspace(-1, 1, num_test_samples),
            y_gt[:, 2].reshape(num_test_samples, num_test_samples), 'w')
    contour(gs[1, 1], np.linspace(-1, 1, num_test_samples), np.linspace(-1, 1, num_test_samples),
            y_gt[:, 3].reshape(num_test_samples, num_test_samples), 'p')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, prefix + "_" + str(i) + "_y_gt.png"))
    plt.close()

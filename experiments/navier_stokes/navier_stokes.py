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
from solver import solve


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ep", "--epochs", type=int, default=30000)
    parser.add_argument("-ntrd", "--num-train-samples-domain", type=int, default=20000)
    parser.add_argument("-rest", "--resample-times", type=int, default=100)
    parser.add_argument("-resn", "--resample-numbers", type=int, default=300)
    parser.add_argument("-nte", "--num-test-samples", type=int, default=51)
    parser.add_argument("-r", "--resample", action="store_true", default=False)
    parser.add_argument("-l", "--load", action="store_true", default=False)
    return parser.parse_known_args()[0]


a = 1
d = 1
Re = 100


def pde(x, u):
    u_vel, v_vel, p = u[:, 0:1], u[:, 1:2], u[:, 2:3]

    u_vel_x = dde.grad.jacobian(u, x, i=0, j=0)
    u_vel_y = dde.grad.jacobian(u, x, i=0, j=1)
    u_vel_t = dde.grad.jacobian(u, x, i=0, j=2)
    u_vel_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)
    u_vel_yy = dde.grad.hessian(u, x, component=0, i=1, j=1)

    v_vel_x = dde.grad.jacobian(u, x, i=1, j=0)
    v_vel_y = dde.grad.jacobian(u, x, i=1, j=1)
    v_vel_t = dde.grad.jacobian(u, x, i=1, j=2)
    v_vel_xx = dde.grad.hessian(u, x, component=1, i=0, j=0)
    v_vel_yy = dde.grad.hessian(u, x, component=1, i=1, j=1)

    p_x = dde.grad.jacobian(u, x, i=2, j=0)
    p_y = dde.grad.jacobian(u, x, i=2, j=1)

    momentum_x = (
        u_vel_t
        + (u_vel * u_vel_x + v_vel * u_vel_y)
        + p_x
        - 1 / Re * (u_vel_xx + u_vel_yy)
    )
    momentum_y = (
        v_vel_t
        + (u_vel * v_vel_x + v_vel * v_vel_y)
        + p_y
        - 1 / Re * (v_vel_xx + v_vel_yy)
    )
    continuity = u_vel_x + v_vel_y

    return [momentum_x, momentum_y, continuity]


def u_func(x):
    return np.where(x[:, 1:2] == 1.0, 1.0, 0.0)


def v_func(x):
    return np.zeros_like(x[:, 0:1])


def plot_loss(loss_train, loss_test):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
    ax.semilogy(1000 * np.arange(len(loss_train)), loss_train, marker='o', label='Training Loss')
    ax.semilogy(1000 * np.arange(len(loss_test)), loss_train, marker='o', label='Testing Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend(loc='best')
    plt.savefig(os.path.join(save_dir, prefix + '_loss.pdf'))
    plt.savefig(os.path.join(save_dir, prefix + '_loss.png'))
    plt.close()


def contour(grid, data_x, data_y, data_z, title, v_min=1, v_max=1, levels=20):
    # plot a contour
    plt.subplot(grid)
    plt.contour(data_x, data_y, data_z, colors='k', linewidths=0.2, levels=levels, vmin=v_min, vmax=v_max)
    plt.contourf(data_x, data_y, data_z, cmap='rainbow', levels=levels, vmin=v_min, vmax=v_max)
    plt.title(title)
    m = plt.cm.ScalarMappable(cmap='rainbow', norm=Normalize(vmin=v_min, vmax=v_max))
    m.set_array(data_z)
    m.set_clim(v_min, v_max)
    cbar = plt.colorbar(m, pad=0.03, aspect=25, format='%.0e')
    cbar.mappable.set_clim(v_min, v_max)


def test_nn(times=None):
    if times is None:
        times = [0.01, 0.5, 1.0]
    for time in times:
        x, y = np.meshgrid(np.linspace(0, 1, num_test_samples), np.linspace(0, 1, num_test_samples))
        X = np.vstack((np.ravel(x), np.ravel(y))).T
        t = time * np.ones(num_test_samples ** 2).reshape(num_test_samples ** 2, 1)
        X = np.hstack((X, t))
        output = model.predict(X)
        u_pred = output[:, 0].reshape(-1)
        v_pred = output[:, 1].reshape(-1)
        p_pred = output[:, 2].reshape(-1)
        u_exact, v_exact, p_exact = solve(num_test_samples, 500, time, Re)
        l2_difference_u = dde.metrics.l2_relative_error(u_exact, u_pred)
        l2_difference_v = dde.metrics.l2_relative_error(v_exact, v_pred)
        # l2_difference_p = dde.metrics.l2_relative_error(p_exact, p_pred)
        residual = np.mean(np.absolute(model.predict(X, operator=pde)))
        print("Accuracy at t = {}:".format(time) + "\\\\")
        print("Mean residual: {:.3f}".format(residual) + "\\\\")
        print("L2 relative error in u: {:.3f}".format(l2_difference_u) + "\\\\")
        print("L2 relative error in v: {:.3f}".format(l2_difference_v) + "\\\\")
        print("L2 relative error: {:.3f}".format((l2_difference_u + l2_difference_v) / 2))

        plt.figure(figsize=(12, 9))
        gs = GridSpec(3, 3)
        u_min = np.min(u_exact)
        u_max = np.max(u_exact)
        v_min = np.min(v_exact)
        v_max = np.max(v_exact)
        p_min = np.min(p_exact)
        p_max = np.max(p_exact)
        contour(gs[0, 0], x, y, u_pred.reshape(num_test_samples, num_test_samples), 'u_pred', u_min, u_max)
        contour(gs[0, 1], x, y, v_pred.reshape(num_test_samples, num_test_samples), 'v_pred', v_min, v_max)
        contour(gs[0, 2], x, y, p_pred.reshape(num_test_samples, num_test_samples), 'p_pred', p_min, p_max)
        contour(gs[1, 0], x, y, u_exact.reshape(num_test_samples, num_test_samples), 'u_exact', u_min, u_max)
        contour(gs[1, 1], x, y, v_exact.reshape(num_test_samples, num_test_samples), 'v_exact', v_min, v_max)
        contour(gs[1, 2], x, y, p_exact.reshape(num_test_samples, num_test_samples), 'p_exact', p_min, p_max)
        error_u = np.abs(u_pred - u_exact) / (np.abs(u_exact) + 1e-6)
        error_v = np.abs(v_pred - v_exact) / (np.abs(v_exact) + 1e-6)
        error_p = np.abs(p_pred - p_exact) / (np.abs(p_exact) + 1e-6)
        contour(gs[2, 0], x, y, error_u.reshape(num_test_samples, num_test_samples), 'u_error', 0, 1)
        contour(gs[2, 1], x, y, error_v.reshape(num_test_samples, num_test_samples), 'v_error', 0, 1)
        contour(gs[2, 2], x, y, error_p.reshape(num_test_samples, num_test_samples), 'p_error', 0, 1)
        plt.savefig(os.path.join(save_dir, prefix + "_t={}.png".format(time)))
        plt.savefig(os.path.join(save_dir, prefix + "_t={}.pdf".format(time)))


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

spatial_domain = dde.geometry.Rectangle([0.0, 0.0], [1.0, 1.0])
temporal_domain = dde.geometry.TimeDomain(0, 1)
spatio_temporal_domain = dde.geometry.GeometryXTime(spatial_domain, temporal_domain)

boundary_condition_u = dde.icbc.DirichletBC(spatio_temporal_domain,
                                            u_func, lambda _, on_boundary: on_boundary, component=0)
boundary_condition_v = dde.icbc.DirichletBC(spatio_temporal_domain,
                                            v_func, lambda _, on_boundary: on_boundary, component=1)

initial_condition_u = dde.icbc.IC(spatio_temporal_domain,
                                  u_func, lambda _, on_initial: on_initial, component=0)
initial_condition_v = dde.icbc.IC(spatio_temporal_domain,
                                  v_func, lambda _, on_initial: on_initial, component=1)

if resample:
    data = dde.data.TimePDE(
        spatio_temporal_domain,
        pde,
        [
            boundary_condition_u,
            boundary_condition_v,
            initial_condition_u,
            initial_condition_v,
        ],
        num_domain=num_train_samples_domain,
        num_boundary=5000,
        num_initial=5000,
        num_test=10000,
    )
else:
    data = dde.data.TimePDE(
        spatio_temporal_domain,
        pde,
        [
            boundary_condition_u,
            boundary_condition_v,
            initial_condition_u,
            initial_condition_v,
        ],
        num_domain=num_train_samples_domain + resample_times * resample_num,
        num_boundary=5000,
        num_initial=5000,
        num_test=10000,
    )

if not load:
    net = dde.nn.FNN([3] + 4 * [50] + [3], "tanh", "Glorot normal")

    model = dde.Model(data, net)

    model.compile("adam", lr=1e-3, loss_weights=[1, 1, 1, 100, 100, 100, 100])
    if resample:
        resampler = dde.callbacks.PDEGradientAccumulativeResampler(period=(epochs // (resample_times + 1) + 1) // 2,
                                                                   sample_num=resample_num, sigma=0.2)
        loss_history, train_state = model.train(epochs=epochs, callbacks=[resampler])
    else:
        loss_history, train_state = model.train(epochs=epochs)
    info = {"net": net, "train_x_all": data.train_x_all, "train_x": data.train_x, "train_x_bc": data.train_x_bc,
            "train_y": data.train_y, "test_x": data.test_x, "test_y": data.test_y,
            "loss_history": loss_history, "train_state": train_state}
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
    loss_history = info["loss_history"]
    train_state = info["train_state"]
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3, loss_weights=[1, 1, 1, 100, 100, 100, 100])

plt.rcParams['font.sans-serif'] = 'Times New Roman'
plt.rcParams.update({'figure.autolayout': True})
plt.rc('font', size=18)
test_nn()
plot_loss(np.array(loss_history.loss_train).sum(axis=1), np.array(loss_history.loss_test).sum(axis=1))

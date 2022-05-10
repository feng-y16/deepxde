"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
import deepxde as dde
import numpy as np
import os
import sys
import argparse
import warnings
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from solver import solve


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ep", "--epochs", type=int, default=20000)
    parser.add_argument("-ntrd", "--num-train-samples-domain", type=int, default=5000)
    parser.add_argument("-rest", "--resample-times", type=int, default=15)
    parser.add_argument("-resn", "--resample-numbers", type=int, default=1000)
    parser.add_argument("-nte", "--num-test-samples", type=int, default=101)
    parser.add_argument("-r", "--resample", action="store_true", default=False)
    parser.add_argument("-l", "--load", nargs='+', default=[])
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
    ax.semilogy(1000 * np.arange(len(loss_train)), loss_train, marker="o", label="Training Loss", linewidth=3)
    ax.semilogy(1000 * np.arange(len(loss_test)), loss_train, marker="o", label="Testing Loss", linewidth=3)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend(loc="best")
    plt.savefig(os.path.join(save_dir, prefix + "_loss.pdf"))
    plt.savefig(os.path.join(save_dir, prefix + "_loss.png"))
    plt.close()


def plot_loss_combined(losses):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
    for legend, loss in losses.items():
        ax.semilogy(1000 * np.arange(len(loss)), loss, marker='o', label=legend, linewidth=3)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Testing Loss")
        ax.legend(loc="best")
    plt.savefig(os.path.join(save_dir, "loss.pdf"))
    plt.savefig(os.path.join(save_dir, "loss.png"))
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


def test_nn(times=None, test_models=None):
    if test_models is None:
        test_models = {}
    if times is None:
        times = [0.5, 1.0]
    test_models_pred_exact = {}
    for legend, test_model in test_models.items():
        test_models_pred_exact[legend] = [None, None, None, None, None, None]
    for time in times:
        x, y = np.meshgrid(np.linspace(0, 1, num_test_samples), np.linspace(0, 1, num_test_samples))
        X = np.vstack((np.ravel(x), np.ravel(y))).T
        t = time * np.ones(num_test_samples ** 2).reshape(num_test_samples ** 2, 1)
        X = np.hstack((X, t))
        u_exact, v_exact, p_exact = solve(num_test_samples, 10000, time, Re)
        num_results = len(models) + 1
        plt.figure(figsize=(12, 3 * num_results))
        gs = GridSpec(num_results, 3)
        u_min = np.min(u_exact)
        u_max = np.max(u_exact)
        v_min = np.min(v_exact)
        v_max = np.max(v_exact)
        p_min = np.min(p_exact)
        p_max = np.max(p_exact)
        result_count = 0
        for legend, test_model in test_models.items():
            output = test_model.predict(X)
            u_pred = output[:, 0].reshape(-1)
            v_pred = output[:, 1].reshape(-1)
            p_pred = output[:, 2].reshape(-1)
            l2_difference_u = dde.metrics.l2_relative_error(u_exact, u_pred)
            l2_difference_v = dde.metrics.l2_relative_error(v_exact, v_pred)
            l2_difference_p = dde.metrics.l2_relative_error(p_exact, p_pred)
            if test_models_pred_exact[legend][0] is None:
                test_models_pred_exact[legend][0] = u_pred
                test_models_pred_exact[legend][1] = v_pred
                test_models_pred_exact[legend][2] = p_pred
                test_models_pred_exact[legend][3] = u_exact
                test_models_pred_exact[legend][4] = v_exact
                test_models_pred_exact[legend][5] = p_exact
            else:
                test_models_pred_exact[legend][0] = np.concatenate((test_models_pred_exact[legend][0], u_pred))
                test_models_pred_exact[legend][1] = np.concatenate((test_models_pred_exact[legend][1], v_pred))
                test_models_pred_exact[legend][2] = np.concatenate((test_models_pred_exact[legend][2], p_pred))
                test_models_pred_exact[legend][3] = np.concatenate((test_models_pred_exact[legend][3], u_exact))
                test_models_pred_exact[legend][4] = np.concatenate((test_models_pred_exact[legend][4], v_exact))
                test_models_pred_exact[legend][5] = np.concatenate((test_models_pred_exact[legend][5], p_exact))
            residual = np.mean(np.absolute(test_model.predict(X, operator=pde)))
            print(legend)
            print("Accuracy at t = {}:".format(time))
            print("Mean residual: {:.3f}".format(residual))
            print("L2 relative error in u, v, p: {:.3f} & {:.3f} & {:.3f}"
                  .format(l2_difference_u, l2_difference_v, l2_difference_p))
            contour(gs[result_count, 0], x, y, u_pred.reshape(num_test_samples, num_test_samples),
                    "u_" + legend, u_min, u_max)
            contour(gs[result_count, 1], x, y, v_pred.reshape(num_test_samples, num_test_samples),
                    "v_" + legend, v_min, v_max)
            contour(gs[result_count, 2], x, y, p_pred.reshape(num_test_samples, num_test_samples),
                    "p_" + legend, p_min, p_max)
            result_count += 1
        contour(gs[-1, 0], x, y, u_exact.reshape(num_test_samples, num_test_samples), 'u_exact', u_min, u_max)
        contour(gs[-1, 1], x, y, v_exact.reshape(num_test_samples, num_test_samples), 'v_exact', v_min, v_max)
        contour(gs[-1, 2], x, y, p_exact.reshape(num_test_samples, num_test_samples), 'p_exact', p_min, p_max)
        plt.savefig(os.path.join(save_dir, "t={}.png".format(time)))
        plt.savefig(os.path.join(save_dir, "t={}.pdf".format(time)))
        plt.close()

    for legend, pred_true in test_models_pred_exact.items():
        u_pred, v_pred, p_pred, u_exact, v_exact, p_exact = pred_true
        l2_difference_u = dde.metrics.l2_relative_error(u_exact, u_pred)
        l2_difference_v = dde.metrics.l2_relative_error(v_exact, v_pred)
        l2_difference_p = dde.metrics.l2_relative_error(p_exact, p_pred)
        print(legend)
        print("L2 relative error in u, v, p: {:.3f} & {:.3f} & {:.3f}"
              .format(l2_difference_u, l2_difference_v, l2_difference_p))


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
    prefix = "LWIS"
else:
    prefix = "PINN"
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
        num_test=100000,
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
        num_test=100000,
    )


plt.rcParams['font.sans-serif'] = 'Times New Roman'
plt.rcParams.update({'figure.autolayout': True})
plt.rc('font', size=18)
models = {}
if len(load) == 0:
    net = dde.nn.FNN([3] + 4 * [50] + [3], "tanh", "Glorot normal")

    model = dde.Model(data, net)

    model.compile("adam", lr=1e-3, loss_weights=[1, 1, 1, 100, 100, 100, 100])
    if resample:
        resampler = dde.callbacks.PDEGradientAccumulativeResampler(period=(epochs // (resample_times + 1) + 1) // 3,
                                                                   sample_num=resample_num, sigma=0.1)
        loss_history, train_state = model.train(epochs=epochs, callbacks=[resampler])
    else:
        loss_history, train_state = model.train(epochs=epochs)
    info = {"net": net, "train_x_all": data.train_x_all, "train_x": data.train_x, "train_x_bc": data.train_x_bc,
            "train_y": data.train_y, "test_x": data.test_x, "test_y": data.test_y,
            "loss_history": loss_history, "train_state": train_state}
    with open(os.path.join(save_dir, prefix + "_info.pkl"), "wb") as f:
        pickle.dump(info, f)
    models[prefix] = model
    load = [prefix]
    plot_loss(np.array(loss_history.loss_train).sum(axis=1), np.array(loss_history.loss_test).sum(axis=1))
else:
    losses_test = {}
    for prefix in load:
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
        models[prefix] = model
        losses_test[prefix] = np.array(loss_history.loss_test).sum(axis=1)
    plot_loss_combined(losses_test)
    test_nn(test_models=models)
    print("draw complete", file=sys.stderr)

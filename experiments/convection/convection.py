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
import tensorflow as tf
import datetime


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ep", "--epochs", type=int, default=50000)
    parser.add_argument("-ntrd", "--num-train-samples-domain", type=int, default=100)
    parser.add_argument("-rest", "--resample-times", type=int, default=4)
    parser.add_argument("-resn", "--resample-numbers", type=int, default=100)
    parser.add_argument("-r", "--resample", action="store_true", default=False)
    parser.add_argument("-l", "--load", nargs='+', default=[])
    return parser.parse_known_args()[0]


def gen_testdata():
    t = np.linspace(0, 1, 100).reshape(-1, 1)
    x = np.linspace(0, 2 * np.pi, 256).reshape(-1, 1)
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = np.zeros((100 * 256, 1))
    return X, y, t, x


def pde(x, y):
    dy_x = dde.grad.jacobian(y, x, i=0, j=0)
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    return dy_t + 100 * dy_x


def plot_loss(loss_train, loss_test):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
    ax.semilogy(epochs // 20 * np.arange(len(loss_train)), loss_train, marker="o", label="Training Loss", linewidth=3)
    ax.semilogy(epochs // 20 * np.arange(len(loss_test)), loss_train, marker="o", label="Testing Loss", linewidth=3)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend(loc="best")
    plt.savefig(os.path.join(save_dir, prefix + "_loss.pdf"))
    plt.savefig(os.path.join(save_dir, prefix + "_loss.png"))
    plt.close()


def plot_loss_combined(losses):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
    for legend, loss in losses.items():
        ax.semilogy(epochs // 20 * np.arange(len(loss)), loss, marker='o', label=legend, linewidth=3)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Testing Loss")
        ax.legend(loc="best")
    plt.savefig(os.path.join(save_dir, "loss.pdf"))
    plt.savefig(os.path.join(save_dir, "loss.png"))
    plt.close()


def func(x):
    sol = np.zeros_like(x[:, 0:1])
    return sol


def test_nn(test_models=None):
    if test_models is None:
        test_models = {}
    num_results = len(models) + 1
    plt.figure(figsize=(12, 3 * num_results))
    gs = GridSpec(num_results, 1)
    X, y_exact, t, x = gen_testdata()
    result_count = 0
    for legend, test_model in test_models.items():
        y_pred = test_model.predict(X)
        pde_pred = test_model.predict(X, operator=pde)
        print(legend)
        print("Mean residual:", np.mean(np.absolute(pde_pred)))
        print("L2 relative error: {:.3f}".format(dde.metrics.l2_relative_error(y_exact, y_pred)))
        top_k = 10
        error = np.abs(y_exact - y_pred).reshape(-1)
        error = error[np.argpartition(-error, top_k)[: top_k]].mean()
        print("Top {:} error: {:.3f}".format(top_k, error))
        ax = plt.subplot(gs[result_count, 0])
        fig = ax.pcolormesh(t * np.ones_like(x.T), np.ones_like(t) * x.T, y_pred.reshape(len(t), len(x)),
                            cmap="rainbow")
        ax.set_xlabel("t")
        ax.set_ylabel("x")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 2 * np.pi)
        ax.set_title("u-" + legend)
        cbar = plt.colorbar(fig, pad=0.05, aspect=10)
        cbar.mappable.set_clim(-1, 1)
        resampled_points = test_model.resampled_data
        if resampled_points is not None:
            resampled_points = np.concatenate(resampled_points, axis=0)
            ax.scatter(resampled_points[:, 1], resampled_points[:, 0], marker=',', s=1, color='y')
        result_count += 1
    ax = plt.subplot(gs[-1, 0])
    fig = ax.pcolormesh(t * np.ones_like(x.T), np.ones_like(t) * x.T, y_exact.reshape(len(t), len(x)), cmap="rainbow")
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 2 * np.pi)
    ax.set_title("u-exact")
    cbar = plt.colorbar(fig, pad=0.05, aspect=10)
    cbar.mappable.set_clim(-1, 1)
    plt.savefig(os.path.join(save_dir, "figure.png"))
    plt.savefig(os.path.join(save_dir, "figure.pdf"))


warnings.filterwarnings("ignore")
print(datetime.datetime.now())
tf.config.threading.set_inter_op_parallelism_threads(1)
args = parse_args()
print(args)
resample = args.resample
resample_times = args.resample_times
resample_num = args.resample_numbers
epochs = args.epochs
num_train_samples_domain = args.num_train_samples_domain
load = args.load
save_dir = os.path.dirname(os.path.abspath(__file__))
if resample:
    prefix = "LWIS"
else:
    prefix = "PINN"
print("resample:", resample)
print("total data points:", num_train_samples_domain + resample_times * resample_num)

geom = dde.geometry.Interval(0, 2 * np.pi)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc = dde.icbc.PeriodicBC(geomtime, 0, lambda _, on_boundary: on_boundary)
ic = dde.icbc.IC(
    geomtime, lambda x: np.sin(x[:, 0:1]), lambda _, on_initial: on_initial
)

if resample:
    data = dde.data.TimePDE(
        geomtime, pde, [bc, ic], num_domain=num_train_samples_domain, num_boundary=500, num_initial=10000
    )
else:
    data = dde.data.TimePDE(
        geomtime, pde, [bc, ic], num_domain=num_train_samples_domain + resample_times * resample_num,
        num_boundary=500, num_initial=10000
    )

plt.rcParams["font.sans-serif"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams.update({"figure.autolayout": True})
plt.rc("font", size=18)
models = {}
if len(load) == 0:
    net = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")
    model = dde.Model(data, net)

    model.compile("adam", lr=1e-3, loss_weights=[1, 100, 100])
    if resample:
        resampler = dde.callbacks.PDEGradientAccumulativeResampler(period=(epochs // (resample_times + 1) + 1) // 3,
                                                                   sample_num=resample_num, sample_count=resample_times,
                                                                   sigma=0.1)
        loss_history, train_state = model.train(epochs=epochs, callbacks=[resampler], display_every=epochs // 20)
    else:
        resampler = None
        loss_history, train_state = model.train(epochs=epochs, display_every=epochs // 20)
    resampled_data = resampler.sampled_train_points if resampler is not None else None
    info = {"net": net, "train_x_all": data.train_x_all, "train_x": data.train_x, "train_x_bc": data.train_x_bc,
            "train_y": data.train_y, "test_x": data.test_x, "test_y": data.test_y,
            "loss_history": loss_history, "train_state": train_state, "resampled_data": resampled_data}
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
        resampled_data = info["resampled_data"]
        model = dde.Model(data, net)
        model.compile("adam", lr=1e-3, loss_weights=[1, 100, 100])
        model.resampled_data = resampled_data
        models[prefix] = model
        losses_test[prefix] = np.array(loss_history.loss_test).sum(axis=1)
    plot_loss_combined(losses_test)
    test_nn(test_models=models)
    print("draw complete", file=sys.stderr)
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
    parser.add_argument("-ntrd", "--num-train-samples-domain", type=int, default=5)
    parser.add_argument("-rest", "--resample-times", type=int, default=2)
    parser.add_argument("-resn", "--resample-numbers", type=int, default=5)
    parser.add_argument("-r", "--resample", action="store_true", default=False)
    parser.add_argument("-l", "--load", nargs='+', default=[])
    return parser.parse_known_args()[0]


r = 200


def ode_system(x, y):
    """ODE system.
    dy/dx = -ry+3r-2rexp(-t)
    """
    dy_x = dde.grad.jacobian(y, x, i=0)
    return dy_x + r * y - 3 * r + 2 * r * tf.exp(-x)


def boundary(_, on_initial):
    return on_initial


def func(x):
    return 3 + (2 * r / (r - 1) - 3) * np.exp(-r * x) - 2 * r / (r - 1) * np.exp(-x)


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


def test_nn(test_models=None):
    if test_models is None:
        test_models = {}
    plt.figure(figsize=(12, 4))
    gs = GridSpec(1, 2)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    x = np.linspace(0, 1, 10000)
    y_exact = func(x.reshape(-1, 1))
    ax1.plot(x, y_exact, label="exact", linewidth=3)
    ax2.plot(x, y_exact, label="exact", linewidth=3)
    result_count = 0
    for legend, test_model in test_models.items():
        y_pred = test_model.predict(x.reshape(-1, 1))
        pde_pred = test_model.predict(x.reshape(-1, 1), operator=ode_system)
        l2_difference_u = dde.metrics.l2_relative_error(y_exact, y_pred)
        print(legend)
        print("Mean residual:", np.mean(np.absolute(pde_pred)))
        print("L2 relative error: {:.3f}".format(l2_difference_u))
        top_k = 10
        error = np.abs(y_exact - y_pred).reshape(-1)
        error = error[np.argpartition(-error, top_k)[: top_k]].mean()
        print("Top {:} error: {:.3f}".format(top_k, error))
        if result_count % 2 == 0:
            ax1.plot(x, y_pred, label=legend, linewidth=3, linestyle="--")
            resampled_points = test_model.resampled_data
            if resampled_points is not None:
                resampled_points = np.concatenate(resampled_points, axis=0)
                ax1.scatter(resampled_points[:, 0], np.zeros_like(resampled_points[:, 0]), marker=',', s=1, color='y')
        else:
            ax2.plot(x, y_pred, label=legend, linewidth=3, linestyle="--")
            resampled_points = test_model.resampled_data
            if resampled_points is not None:
                resampled_points = np.concatenate(resampled_points, axis=0)
                ax2.scatter(resampled_points[:, 0], np.zeros_like(resampled_points[:, 0]), marker=',', s=1, color='y')
        result_count += 1
    ax1.set_xlabel("t")
    ax1.set_title("u")
    ax1.legend(loc="best")
    ax2.set_xlabel("t")
    ax2.set_title("u")
    ax2.legend(loc="best")
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
print("resample:", resample)
print("total data points:", num_train_samples_domain + resample_times * resample_num)
if resample:
    prefix = "LWIS"
else:
    prefix = "PINN"

geom = dde.geometry.TimeDomain(0, 1)
ic = dde.icbc.IC(geom, lambda x: 0, boundary, component=0)

if resample:
    data = dde.data.PDE(geom, ode_system, [ic], num_train_samples_domain, 2, solution=func, num_test=1000)
else:
    data = dde.data.PDE(geom, ode_system, [ic], num_train_samples_domain + resample_times * resample_num,
                        2, solution=func, num_test=1000)

plt.rcParams["font.sans-serif"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams.update({"figure.autolayout": True})
plt.rc("font", size=18)
models = {}
if len(load) == 0:
    layer_size = [1] + [50] * 3 + [1]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.nn.FNN(layer_size, activation, initializer)

    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3, metrics=["l2 relative error"], loss_weights=[1, 100])
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
        model.compile("adam", lr=1e-3, metrics=["l2 relative error"], loss_weights=[1, 100])
        model.resampled_data = resampled_data
        models[prefix] = model
        losses_test[prefix] = np.array(loss_history.loss_test).sum(axis=1)
    plot_loss_combined(losses_test)
    test_nn(test_models=models)
    print("draw complete", file=sys.stderr)
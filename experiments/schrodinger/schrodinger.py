"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
import pdb

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ep", "--epochs", type=int, default=20000)
    parser.add_argument("-ntrd", "--num-train-samples-domain", type=int, default=20000)
    parser.add_argument("-rest", "--resample-times", type=int, default=10)
    parser.add_argument("-resn", "--resample-numbers", type=int, default=1000)
    parser.add_argument("-r", "--resample", action="store_true", default=False)
    parser.add_argument("-l", "--load", nargs='+', default=[])
    parser.add_argument("-d", "--dimension", type=int, default=5)
    return parser.parse_known_args()[0]


c = 1


def pde_d(dimension, x, y):
    laplace = None
    for i in range(dimension):
        if laplace is None:
            laplace = dde.grad.hessian(y, x, i=i, j=i)
        else:
            laplace += dde.grad.hessian(y, x, i=i, j=i)
    psi_x = tf.zeros_like(x[:, 0:1])
    for i in range(dimension):
        psi_x += tf.cos(x[i:(i + 1)])
    psi_x *= 2 / dimension
    psi_x = -1 / c ** 2 * tf.exp(psi_x)
    for i in range(dimension):
        psi_x += tf.sin(x[i:(i + 1)]) ** 2 / dimension ** 2 - tf.cos(x[i:(i + 1)]) / dimension
    psi_x -= 3
    return laplace - y ** 3 - psi_x * y - 3 * y


def func_d(dimension, x):
    sol = np.zeros_like(x[:, 0:1])
    for i in range(dimension):
        sol += np.cos(x[:, i:(i + 1)])
    sol /= dimension
    sol = np.exp(sol) / c
    return sol


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


def test_nn(test_models=None):
    if test_models is None:
        test_models = {}
    X = test_models["PINN"].data.test_x
    y_exact = func(X)
    for legend, test_model in test_models.items():
        y_pred = test_model.predict(X)
        pde_pred = test_model.predict(X, operator=pde)
        print(legend)
        print("Mean residual:", np.mean(np.absolute(pde_pred)))
        print("L2 relative error:", dde.metrics.l2_relative_error(y_exact, y_pred))


warnings.filterwarnings("ignore")
args = parse_args()
resample = args.resample
resample_times = args.resample_times
resample_num = args.resample_numbers
epochs = args.epochs
num_train_samples_domain = args.num_train_samples_domain
load = args.load
d = args.dimension


def pde(x, y):
    return pde_d(d, x, y)


def func(x):
    return func_d(d, x)


save_dir = os.path.dirname(os.path.abspath(__file__))
if resample:
    prefix = "LWIS"
else:
    prefix = "PINN"
print("resample:", resample)
print("total data points:", num_train_samples_domain + resample_times * resample_num)

geom = dde.geometry.Hypercube([0 for _ in range(d)], [2 * np.pi for _ in range(d)])

bc = dde.icbc.DirichletBC(geom, func, lambda _, on_boundary: on_boundary)

if resample:
    data = dde.data.PDE(geom, pde, [bc], num_domain=num_train_samples_domain, num_boundary=10000)
else:
    data = dde.data.PDE(geom, pde, [bc], num_domain=num_train_samples_domain + resample_times * resample_num,
                        num_boundary=10000)

plt.rcParams['font.sans-serif'] = 'Times New Roman'
plt.rcParams.update({'figure.autolayout': True})
plt.rc('font', size=18)
models = {}
if len(load) == 0:
    net = dde.nn.FNN([d] + [20] * 3 + [1], "tanh", "Glorot normal")
    model = dde.Model(data, net)

    model.compile("adam", lr=1e-3, loss_weights=[1, 100])
    if resample:
        resampler = dde.callbacks.PDEGradientAccumulativeResampler(period=(epochs // (resample_times + 1) + 1) // 3,
                                                                   sample_num=resample_num, sigma=1)
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
        model.compile("adam", lr=1e-3, loss_weights=[1, 100])
        models[prefix] = model
        losses_test[prefix] = np.array(loss_history.loss_test).sum(axis=1)
    plot_loss_combined(losses_test)
    test_nn(test_models=models)
    print("draw complete", file=sys.stderr)

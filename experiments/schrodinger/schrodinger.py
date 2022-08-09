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
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--num-train-samples-domain", type=int, default=3000)
    parser.add_argument("--num-train-samples-boundary", type=int, default=600)
    parser.add_argument("--num-train-samples-initial", type=int, default=0)
    parser.add_argument("--resample-ratio", type=float, default=0.4)
    parser.add_argument("--resample-times", type=int, default=4)
    parser.add_argument("--resample-splits", type=int, default=2)
    parser.add_argument("--resample", action="store_true", default=False)
    parser.add_argument("--adversarial", action="store_true", default=False)
    parser.add_argument("--annealing", action="store_true", default=False)
    parser.add_argument("--loss-weights", nargs="+", type=float, default=[1, 100])
    parser.add_argument("--load", nargs='+', default=[])
    parser.add_argument("--dimension", type=int, default=6)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--sensitivity", action="store_true", default=False)
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
        psi_x += tf.cos(x[:, i:(i + 1)])
    psi_x *= 2 / dimension
    psi_x = -1 / c ** 2 * tf.exp(psi_x)
    for i in range(dimension):
        psi_x += tf.sin(x[:, i:(i + 1)]) ** 2 / dimension ** 2 - tf.cos(x[:, i:(i + 1)]) / dimension
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
        if legend not in ["PINN_10000", "LWIS_10000_0.1"]:
            continue
        ax.semilogy(epochs // 20 * np.arange(len(loss)), loss, marker='o', label=legend[:4], linewidth=3)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Testing Loss")
        ax.legend(loc="best")
    plt.savefig(os.path.join(save_dir, "loss.pdf"))
    plt.savefig(os.path.join(save_dir, "loss.png"))
    plt.close()


def test_nn_sensitivity(test_models=None, losses=None):
    if test_models is None:
        test_models = {}
    if losses is None:
        losses = {}
    X = test_models[list(test_models.keys())[0]].data.test_x
    y_exact = func(X)
    model_errors = dict()
    model_top_k_errors = dict()
    model_losses = dict()
    top_k = 10
    for legend, test_model in test_models.items():
        y_pred = test_model.predict(X)
        l2_difference_u = dde.metrics.l2_relative_error(y_exact, y_pred)
        error = np.abs(y_exact - y_pred).reshape(-1)
        error = error[np.argpartition(-error, top_k)[: top_k]].mean()
        print("{:} & {:.3f} & {:.3f}\\\\".format(legend, l2_difference_u, error))
        parsed_legend = legend.split("_")
        if parsed_legend[0] not in model_errors.keys():
            model_errors[parsed_legend[0]] = dict()
            model_top_k_errors[parsed_legend[0]] = dict()
            model_losses[parsed_legend[0]] = dict()
        if parsed_legend[0] != "LWIS" and parsed_legend[0] != "LWIS-A":
            model_errors[parsed_legend[0]] = l2_difference_u
            model_top_k_errors[parsed_legend[0]] = error
            model_losses[parsed_legend[0]] = losses[legend]
        else:
            num_splits = int(parsed_legend[1])
            LWIS_sigma = float(parsed_legend[2])
            if LWIS_sigma not in model_errors[parsed_legend[0]].keys():
                model_errors[parsed_legend[0]][LWIS_sigma] = dict()
                model_top_k_errors[parsed_legend[0]][LWIS_sigma] = dict()
            model_errors[parsed_legend[0]][LWIS_sigma][num_splits] = l2_difference_u
            model_top_k_errors[parsed_legend[0]][LWIS_sigma][num_splits] = error
            if num_splits not in model_losses[parsed_legend[0]].keys():
                model_losses[parsed_legend[0]][num_splits] = dict()
            model_losses[parsed_legend[0]][num_splits][LWIS_sigma] = losses[legend]
    plt.figure(figsize=(12, 4))
    gs = GridSpec(1, 2)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    for prefix in model_errors.keys():
        if prefix != "LWIS" and prefix != "LWIS-A":
            pass
        else:
            for LWIS_sigma, LWIS_error in model_errors[prefix].items():
                ax1.plot(list(LWIS_error.keys()), list(LWIS_error.values()), marker="o",
                             label=r"LWIS-$\sigma={:.2f}$".format(LWIS_sigma), linewidth=3)
    ax1.set_xlabel("Number of Splits")
    ax1.set_ylabel(r"$l_2$ Relative Error")
    ax1.legend(loc="best")
    for prefix in model_top_k_errors.keys():
        if prefix != "LWIS" and prefix != "LWIS-A":
            pass
        else:
            for LWIS_sigma, LWIS_top_k_error in model_top_k_errors[prefix].items():
                ax2.plot(list(LWIS_top_k_error.keys()), list(LWIS_top_k_error.values()), marker="o",
                             label=r"LWIS-$\sigma={:.2f}$".format(LWIS_sigma), linewidth=3)
    ax2.set_xlabel("Number of Splits")
    ax2.set_ylabel("Top {:} Error".format(top_k))
    ax2.legend(loc="best")
    plt.savefig(os.path.join(save_dir, "sensitivity.pdf"))
    plt.savefig(os.path.join(save_dir, "sensitivity.png"))
    plt.close()
    plt.figure(figsize=(6, 4))


def test_nn(test_models=None):
    if test_models is None:
        test_models = {}
    X = test_models[list(test_models.keys())[0]].data.test_x
    y_exact = func(X)
    for legend, test_model in test_models.items():
        y_pred = test_model.predict(X)
        l2_difference_u = dde.metrics.l2_relative_error(y_exact, y_pred)
        top_k = 10
        error = np.abs(y_exact - y_pred).reshape(-1)
        error = error[np.argpartition(-error, top_k)[: top_k]].mean()
        print("{:} & {:.3f} & {:.3f}\\\\".format(legend, l2_difference_u, error))


warnings.filterwarnings("ignore")
print(datetime.datetime.now())
tf.config.threading.set_inter_op_parallelism_threads(1)
args = parse_args()
if args.sensitivity:
    args.args.num_train_samples_domain *= 2
    args.args.num_train_samples_boundary *= 2
    args.args.num_train_samples_initial *= 2
print(args)
resample = args.resample
adversarial = args.adversarial
annealing = args.annealing
loss_weights = args.loss_weights
resample_times = args.resample_times
resample_ratio = args.resample_ratio
resample_splits = args.resample_splits
epochs = args.epochs
num_train_samples_domain = args.num_train_samples_domain
num_train_samples_boundary = args.num_train_samples_boundary
num_train_samples_initial = args.num_train_samples_initial
load = args.load
save_dir = os.path.dirname(os.path.abspath(__file__))
d = args.dimension
sigma = args.sigma
if resample:
    prefix = "LWIS"
elif adversarial:
    prefix = "AT"
else:
    prefix = "PINN"
if annealing:
    prefix += "-A"
if prefix[:4] == "LWIS" and args.sensitivity:
    prefix += "_{:}_{:}".format(resample_splits, sigma)
print("resample:", resample, resample_splits)
print("adversarial:", adversarial)
print("annealing:", annealing)
print("data points:", num_train_samples_domain, num_train_samples_boundary, num_train_samples_initial)


def pde(x, y):
    return pde_d(d, x, y)


def func(x):
    return func_d(d, x)


geom = dde.geometry.Hypercube([0 for _ in range(d)], [2 * np.pi for _ in range(d)])

bc = dde.icbc.DirichletBC(geom, func, lambda _, on_boundary: on_boundary)

if resample or adversarial:
    data = dde.data.PDE(
        geom, pde, [bc],
        num_domain=int(num_train_samples_domain - num_train_samples_domain * resample_ratio),
        num_boundary=int(num_train_samples_boundary - num_train_samples_boundary * resample_ratio),
        num_test=20000)
else:
    data = dde.data.PDE(
        geom, pde, [bc],
        num_domain=num_train_samples_domain,
        num_boundary=num_train_samples_boundary,
        num_test=20000)

plt.rcParams["font.sans-serif"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams.update({"figure.autolayout": True})
plt.rc("font", size=18)
models = {}
if len(load) == 0:
    net = dde.nn.FNN([d] + [50] * 5 + [1], "tanh", "Glorot normal")
    model = dde.Model(data, net)

    model.compile("adam", lr=1e-3, loss_weights=loss_weights)
    callbacks = []
    if resample:
        resampler = dde.callbacks.PDELossAccumulativeResampler(
            sample_every=(epochs // (resample_times + 1) + 1) // 3,
            sample_num_domain=int(num_train_samples_domain * resample_ratio),
            sample_num_boundary=int(num_train_samples_boundary * resample_ratio),
            sample_num_initial=int(num_train_samples_initial * resample_ratio),
            sample_times=resample_times, sigma=sigma,
            sample_splits=resample_splits)
        callbacks.append(resampler)
    elif adversarial:
        resampler = dde.callbacks.PDEAdversarialAccumulativeResampler(
            sample_every=(epochs // (resample_times + 1) + 1) // 3,
            sample_num_domain=int(num_train_samples_domain * resample_ratio),
            sample_num_boundary=int(num_train_samples_boundary * resample_ratio),
            sample_num_initial=int(num_train_samples_initial * resample_ratio),
            sample_times=resample_times, eta=0.01)
        callbacks.append(resampler)
    if annealing:
        resampler = dde.callbacks.PDELearningRateAnnealing(adjust_every=epochs // 20, loss_weights=loss_weights)
        callbacks.append(resampler)
    loss_history, train_state = model.train(epochs=epochs, callbacks=callbacks, display_every=epochs // 20)
    resampled_data = callbacks[0].sampled_train_points if len(callbacks) > 0 and prefix[:4] != "PINN" else None
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
        try:
            with open(os.path.join(save_dir, prefix + "_info.pkl"), "rb") as f:
                info = pickle.load(f)
        except FileNotFoundError:
            continue
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
        model.compile("adam", lr=1e-3, loss_weights=loss_weights)
        model.resampled_data = resampled_data
        models[prefix] = model
        losses_test[prefix] = np.array(loss_history.loss_test).sum(axis=1)
    plot_loss_combined(losses_test)
    if args.sensitivity:
        test_nn_sensitivity(test_models=models, losses=losses_test)
    else:
        test_nn(test_models=models)
    print("draw complete", file=sys.stderr)

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
    parser.add_argument("--num-train-samples-domain", type=int, default=4000)
    parser.add_argument("--num-train-samples-boundary", type=int, default=100)
    parser.add_argument("--num-train-samples-initial", type=int, default=100)
    parser.add_argument("--resample-ratio", type=float, default=0.5)
    parser.add_argument("--resample-every", type=int, default=1)
    parser.add_argument("--resample", action="store_true", default=False)
    parser.add_argument("--adversarial", action="store_true", default=False)
    parser.add_argument("--annealing", action="store_true", default=False)
    parser.add_argument("--loss-weights", nargs="+", type=float, default=[1, 100, 100])
    parser.add_argument("--load", nargs='+', default=[])
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--draw-annealing", action="store_true", default=False)
    parser.add_argument("--sensitivity", action="store_true", default=False)
    return parser.parse_known_args()[0]


def gen_testdata():
    burgers_data = np.load(os.path.join(save_dir, "Burgers.npz"))
    t, x, exact = burgers_data["t"], burgers_data["x"], burgers_data["usol"].T
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = exact.flatten()[:, None]
    return X, y, t, x


def pde(x, y):
    dy_x = dde.grad.jacobian(y, x, i=0, j=0)
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t + y * dy_x - 0.01 / np.pi * dy_xx


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


def plot_loss_combined(losses, filename):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
    for legend, loss in losses.items():
        if legend[-2:] == "-A":
            continue
        ax.semilogy(epochs // 20 * np.arange(len(loss)), loss, marker='o', label=legend, linewidth=3)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Testing Loss")
    ax.legend(loc="best")
    plt.savefig(os.path.join(save_dir, "{:}.pdf".format(filename)))
    plt.savefig(os.path.join(save_dir, "{:}.png".format(filename)))
    plt.close()


def test_nn_sensitivity(test_models=None, losses=None):
    if test_models is None:
        test_models = {}
    if losses is None:
        losses = {}
    X, y_exact, t, x = gen_testdata()
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
        if parsed_legend[0] != "LWIS":
            continue
        else:
            num_splits = int(parsed_legend[1])
            model_errors[num_splits] = l2_difference_u
            model_top_k_errors[num_splits] = error
            model_losses[num_splits] = losses[legend]
    plt.figure(figsize=(6, 5))
    gs = GridSpec(1, 1)
    ax1 = plt.subplot(gs[0, 0])
    ax1.bar([str(key) for key in model_errors.keys()], model_errors.values())
    ax1.set_xlabel(r"$K$")
    ax1.set_ylabel(r"$l_2$ Relative Error")
    plt.savefig(os.path.join(save_dir, "sensitivity1.pdf"))
    plt.savefig(os.path.join(save_dir, "sensitivity1.png"))
    plt.close()
    plt.figure(figsize=(6, 5))
    gs = GridSpec(1, 1)
    ax2 = plt.subplot(gs[0, 0])
    ax2.bar([str(key) for key in model_top_k_errors.keys()], model_top_k_errors.values())
    ax2.set_xlabel(r"$K$")
    ax2.set_ylabel("Top {:} Error".format(top_k))
    plt.savefig(os.path.join(save_dir, "sensitivity2.pdf"))
    plt.savefig(os.path.join(save_dir, "sensitivity2.png"))
    plt.close()


def test_nn(test_models=None, draw_annealing=False):
    if test_models is None:
        test_models = {}
    num_results = len(test_models)
    if not draw_annealing:
        num_results //= 2
        num_results = max(num_results, 1)
    num_results += 1
    plt.figure(figsize=(12, 3 * num_results))
    gs = GridSpec(num_results, 1)
    X, y_exact, t, x = gen_testdata()
    result_count = 0
    for legend, test_model in test_models.items():
        y_pred = test_model.predict(X)
        l2_difference_u = dde.metrics.l2_relative_error(y_exact, y_pred)
        top_k = 10
        error = np.abs(y_exact - y_pred).reshape(-1)
        error = error[np.argpartition(-error, top_k)[: top_k]].mean()
        print("    {:} & {:.3f} & {:.3f}\\\\".format(legend, l2_difference_u, error))
        if not draw_annealing and legend.split("_")[0][-2:] == "-A":
            continue
        ax = plt.subplot(gs[result_count, 0])
        fig = ax.pcolormesh(t * np.ones_like(x.T), np.ones_like(t) * x.T, y_pred.reshape(len(t), len(x)),
                            cmap="rainbow")
        ax.set_xlabel("t")
        ax.set_ylabel("x")
        ax.set_xlim(0, 1)
        ax.set_ylim(-1, 1)
        ax.set_title("u-" + legend.split("_")[0])
        cbar = plt.colorbar(fig, pad=0.05, aspect=10)
        cbar.mappable.set_clim(-1, 1)
        resampled_points = test_model.resampled_data
        if resampled_points is not None:
            resampled_points = np.concatenate(resampled_points, axis=0)
            ax.scatter(resampled_points[-50000:, 1], resampled_points[-50000:, 0], marker='X', s=0.1, color='black')
        result_count += 1
    ax = plt.subplot(gs[-1, 0])
    fig = ax.pcolormesh(t * np.ones_like(x.T), np.ones_like(t) * x.T, y_exact.reshape(len(t), len(x)), cmap="rainbow")
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 1)
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
adversarial = args.adversarial
annealing = args.annealing
sensitivity = args.sensitivity
loss_weights = args.loss_weights
resample_ratio = args.resample_ratio
resample_every = args.resample_every
epochs = args.epochs
num_train_samples_domain = args.num_train_samples_domain
num_train_samples_boundary = args.num_train_samples_boundary
num_train_samples_initial = args.num_train_samples_initial
load = args.load
save_dir = os.path.dirname(os.path.abspath(__file__))
if resample:
    prefix = "LWIS"
elif adversarial:
    prefix = "AT"
else:
    prefix = "PINN"
if domain_only:
    prefix += "-D"
if boundary_only:
    prefix += "-B"
if annealing:
    prefix += "-A"
if prefix[:4] == "LWIS" and sensitivity:
    prefix += "_{:}".format(resample_every)
print("resample:", resample, resample_times, resample_every)
print("adversarial:", adversarial)
print("annealing:", annealing)
print("data points:", num_train_samples_domain, num_train_samples_boundary, num_train_samples_initial)

geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 1.0)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc = dde.icbc.DirichletBC(geomtime, lambda x: np.zeros_like(x[:, 0:1]), lambda _, on_boundary: on_boundary)
ic = dde.icbc.IC(
    geomtime, lambda x: -np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial
)

if resample or adversarial:
    data = dde.data.TimePDE(
        geomtime, pde, [bc, ic],
        num_domain=int(num_train_samples_domain - num_train_samples_domain * resample_ratio),
        num_boundary=num_train_samples_boundary,
        num_initial=num_train_samples_initial
    )
else:
    data = dde.data.TimePDE(
        geomtime, pde, [bc, ic],
        num_domain=num_train_samples_domain,
        num_boundary=num_train_samples_boundary,
        num_initial=num_train_samples_initial
    )

plt.rcParams["font.sans-serif"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams.update({"figure.autolayout": True})
plt.rc("font", size=18)
models = {}
if len(load) == 0:
    if os.path.isfile(os.path.join(save_dir, prefix + "_info.pkl")):
        print("skipping")
        exit(0)
    net = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")
    model = dde.Model(data, net)

    model.compile("adam", lr=1e-3, loss_weights=loss_weights)
    callbacks = []
    if resample:
        sample_num_domain = int(num_train_samples_domain * resample_ratio)
        resampler = dde.callbacks.PDELossAccumulativeResampler(
            sample_every=resample_every, sample_num_domain=sample_num_domain)
        callbacks.append(resampler)
    elif adversarial:
        resampler = dde.callbacks.PDEAdversarialAccumulativeResampler(
            sample_every=epochs // resample_times,
            sample_num_domain=num_train_samples_domain,
            sample_num_boundary=0,
            sample_num_initial=0,
            sample_times=resample_times,
            eta=0.01)
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
    if sensitivity:
        plot_loss_combined(losses_test, "loss_sensitivity")
        test_nn_sensitivity(test_models=models, losses=losses_test)
    else:
        plot_loss_combined(losses_test, "loss")
        test_nn(test_models=models, draw_annealing=args.draw_annealing)
    print("draw complete", file=sys.stderr)

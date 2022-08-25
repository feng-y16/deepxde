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
from matplotlib.patches import ConnectionPatch
import tensorflow as tf
import datetime


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--num-train-samples-domain", type=int, default=20)
    parser.add_argument("--num-train-samples-boundary", type=int, default=0)
    parser.add_argument("--num-train-samples-initial", type=int, default=2)
    parser.add_argument("--resample-ratio", type=float, default=0.4)
    parser.add_argument("--resample-times", type=int, default=2)
    parser.add_argument("--resample-splits", type=int, default=1)
    parser.add_argument("--resample", action="store_true", default=False)
    parser.add_argument("--adversarial", action="store_true", default=False)
    parser.add_argument("--annealing", action="store_true", default=False)
    parser.add_argument("--loss-weights", nargs="+", type=float, default=[1, 100])
    parser.add_argument("--load", nargs='+', default=[])
    parser.add_argument("--draw-annealing", action="store_true", default=False)
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


def zone_and_linked(ax, axins, zone_left, zone_right, x, y, linked='bottom',
                    x_ratio=0.1, y_ratio=0.3):
    # references: https://blog.csdn.net/weixin_45826022/article/details/113486448
    """缩放内嵌图形，并且进行连线
    ax:         调用plt.subplots返回的画布。例如： fig,ax = plt.subplots(1,1)
    axins:      内嵌图的画布。 例如 axins = ax.inset_axes((0.4,0.1,0.4,0.3))
    zone_left:  要放大区域的横坐标左端点
    zone_right: 要放大区域的横坐标右端点
    x:          X轴标签
    y:          列表，所有y值
    linked:     进行连线的位置，{'bottom','top','left','right'}
    x_ratio:    X轴缩放比例
    y_ratio:    Y轴缩放比例
    """
    xlim_left = x[zone_left] - (x[zone_right] - x[zone_left]) * x_ratio
    xlim_right = x[zone_right] + (x[zone_right] - x[zone_left]) * x_ratio

    y_data = np.hstack([yi[zone_left:zone_right] for yi in y])
    ylim_bottom = np.min(y_data[:, 0]) - (np.max(y_data[:, 0]) - np.min(y_data[:, 0])) * y_ratio
    ylim_top = np.max(y_data[:, 0]) + (np.max(y_data[:, 0]) - np.min(y_data[:, 0])) * y_ratio

    axins.set_xlim(xlim_left, xlim_right)
    axins.set_ylim(ylim_bottom, ylim_top)

    ax.plot([xlim_left, xlim_right, xlim_right, xlim_left, xlim_left],
            [ylim_bottom, ylim_bottom, ylim_top, ylim_top, ylim_bottom], "black")

    if linked == 'bottom':
        xyA_1, xyB_1 = (xlim_left, ylim_top), (xlim_left, ylim_bottom)
        xyA_2, xyB_2 = (xlim_right, ylim_top), (xlim_right, ylim_bottom)
    elif linked == 'top':
        xyA_1, xyB_1 = (xlim_left, ylim_bottom), (xlim_left, ylim_top)
        xyA_2, xyB_2 = (xlim_right, ylim_bottom), (xlim_right, ylim_top)
    elif linked == 'left':
        xyA_1, xyB_1 = (xlim_right, ylim_top), (xlim_left, ylim_top)
        xyA_2, xyB_2 = (xlim_right, ylim_bottom), (xlim_left, ylim_bottom)
    elif linked == 'right':
        xyA_1, xyB_1 = (xlim_left, ylim_top), (xlim_right, ylim_top)
        xyA_2, xyB_2 = (xlim_left, ylim_bottom), (xlim_right, ylim_bottom)

    con = ConnectionPatch(xyA=xyA_1, xyB=xyB_1, coordsA="data",
                          coordsB="data", axesA=axins, axesB=ax)
    axins.add_artist(con)
    con = ConnectionPatch(xyA=xyA_2, xyB=xyB_2, coordsA="data",
                          coordsB="data", axesA=axins, axesB=ax)
    axins.add_artist(con)


def test_nn(test_models=None, draw_annealing=False):
    if test_models is None:
        test_models = {}
    num_results = len(test_models)
    if not draw_annealing:
        num_results //= 2
    plt.figure(figsize=(num_results * 5, 4))
    gs = GridSpec(1, num_results)
    x = np.linspace(0, 1, 10000)
    y_exact = func(x.reshape(-1, 1))
    result_count = 0
    plot_index = 0
    for legend, test_model in test_models.items():
        y_pred = test_model.predict(x.reshape(-1, 1))
        l2_difference_u = dde.metrics.l2_relative_error(y_exact, y_pred)
        top_k = 10
        error = np.abs(y_exact - y_pred).reshape(-1)
        error = error[np.argpartition(-error, top_k)[: top_k]].mean()
        print("{:} & {:.3f} & {:.3f}\\\\".format(legend, l2_difference_u, error))
        if not draw_annealing and legend.split("_")[0][-2:] == "-A":
            continue
        ax = plt.subplot(gs[0, plot_index])
        ax.plot(x, y_exact, label="exact", linewidth=3)
        ax.plot(x, y_pred, label=legend, linewidth=3, linestyle="--")
        ax_insert = ax.inset_axes((0.63, 0.4, 0.3, 0.3))
        ax_insert.plot(x, y_exact, label="exact", linewidth=3)
        ax_insert.plot(x, y_pred, label=legend, linewidth=3, linestyle="--")
        zone_and_linked(ax, ax_insert, 50, 500, x, [y_exact, y_pred], "right")
        ax_insert.set_xticks([])
        ax_insert.set_yticks([])
        resampled_points = test_model.resampled_data
        if resampled_points is not None:
            resampled_points = np.concatenate(resampled_points, axis=0)
            ax.scatter(resampled_points[:, 0], np.zeros_like(resampled_points[:, 0]), marker='X', s=10, color='black')
        ax.set_xlabel("t")
        ax.set_title("u")
        ax.legend(loc="best")
        result_count += 1
        plot_index += 1
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
loss_weights = args.loss_weights
resample_times = args.resample_times
resample_splits = args.resample_splits
resample_ratio = args.resample_ratio
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
if annealing:
    prefix += "-A"
print("resample:", resample, resample_splits)
print("adversarial:", adversarial)
print("annealing:", annealing)
print("data points:", num_train_samples_domain, num_train_samples_boundary, num_train_samples_initial)

geom = dde.geometry.TimeDomain(0, 1)
ic = dde.icbc.IC(geom, lambda x: np.zeros_like(x[:, 0:1]), boundary, component=0)

if resample or adversarial:
    data = dde.data.PDE(
        geom, ode_system, [ic],
        num_domain=int(num_train_samples_domain - num_train_samples_domain * resample_ratio),
        num_boundary=num_train_samples_initial,
        solution=func, num_test=1000)
else:
    data = dde.data.PDE(
        geom, ode_system, [ic],
        num_domain=num_train_samples_domain,
        num_boundary=num_train_samples_initial,
        solution=func, num_test=1000)

plt.rcParams["font.sans-serif"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams.update({"figure.autolayout": True})
plt.rc("font", size=18)
models = {}
if len(load) == 0:
    layer_size = [1] + [20] * 3 + [1]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.nn.FNN(layer_size, activation, initializer)

    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3, metrics=["l2 relative error"], loss_weights=loss_weights)
    callbacks = []
    if resample:
        resampler = dde.callbacks.PDELossAccumulativeResampler(
            sample_every=(epochs // (resample_times + 1) + 1) // 3,
            sample_num_domain=int(num_train_samples_domain * resample_ratio),
            sample_num_boundary=0,
            sample_num_initial=0,
            sample_times=resample_times,
            sample_splits=resample_splits)
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
        model.compile("adam", lr=1e-3, metrics=["l2 relative error"], loss_weights=loss_weights)
        model.resampled_data = resampled_data
        models[prefix] = model
        losses_test[prefix] = np.array(loss_history.loss_test).sum(axis=1)
    plot_loss_combined(losses_test)
    test_nn(test_models=models, draw_annealing=args.draw_annealing)
    print("draw complete", file=sys.stderr)

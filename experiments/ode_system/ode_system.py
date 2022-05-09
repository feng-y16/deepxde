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
    parser.add_argument("-ntrd", "--num-train-samples-domain", type=int, default=20)
    parser.add_argument("-rest", "--resample-times", type=int, default=20)
    parser.add_argument("-resn", "--resample-numbers", type=int, default=4)
    parser.add_argument("-r", "--resample", action="store_true", default=False)
    parser.add_argument("-l", "--load", nargs='+', default=[])
    return parser.parse_known_args()[0]


def ode_system(x, y):
    """ODE system.
    dy1/dx = y2
    dy2/dx = -y1
    """
    y1, y2 = y[:, 0:1], y[:, 1:]
    dy1_x = dde.grad.jacobian(y, x, i=0)
    dy2_x = dde.grad.jacobian(y, x, i=1)
    return [dy1_x - y2, dy2_x + y1]


def boundary(_, on_initial):
    return on_initial


def func(x):
    """
    y1 = sin(x)
    y2 = cos(x)
    """
    return np.hstack((np.sin(x), np.cos(x)))


def plot_loss(loss_train, loss_test):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
    ax.semilogy(1000 * np.arange(len(loss_train)), loss_train, marker='o', label='Training Loss', linewidth=3)
    ax.semilogy(1000 * np.arange(len(loss_test)), loss_train, marker='o', label='Testing Loss', linewidth=3)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend(loc='best')
    plt.savefig(os.path.join(save_dir, prefix + '_loss.pdf'))
    plt.savefig(os.path.join(save_dir, prefix + '_loss.png'))
    plt.close()


def plot_loss_combined(losses):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
    for legend, loss in losses.items():
        ax.semilogy(1000 * np.arange(len(loss)), loss, marker='o', label=legend, linewidth=3)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Testing Loss')
        ax.legend(loc='best')
    plt.savefig(os.path.join(save_dir, 'loss.pdf'))
    plt.savefig(os.path.join(save_dir, 'loss.png'))
    plt.close()


def test_nn(test_models=None):
    if test_models is None:
        test_models = {}
    plt.figure(figsize=(12, 4))
    gs = GridSpec(1, 2)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    x = np.linspace(0, 10, 1000)
    y_true = func(x.reshape(-1, 1))
    line_styles = [":", "--"]
    result_count = 0
    for legend, test_model in test_models.items():
        y_pred = test_model.predict(x.reshape(-1, 1))
        pde_pred = test_model.predict(x.reshape(-1, 1), operator=ode_system)
        print(legend)
        print("Mean residual:", np.mean(np.absolute(pde_pred)))
        print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))
        ax1.plot(x, y_pred[:, 0], label=legend, linewidth=3, linestyle=line_styles[result_count % 2])
        ax2.plot(x, y_pred[:, 1], label=legend, linewidth=3, linestyle=line_styles[result_count % 2])
        result_count += 1
    ax1.plot(x, y_true[:, 0], label="exact", linewidth=3)
    ax1.set_title("u1")
    ax1.legend(loc="best")
    ax2.plot(x, y_true[:, 1], label="exact", linewidth=3)
    ax2.set_title("u2")
    ax2.legend(loc="best")
    plt.savefig(os.path.join(save_dir, "figure.png"))
    plt.savefig(os.path.join(save_dir, "figure.pdf"))


warnings.filterwarnings("ignore")
args = parse_args()
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

geom = dde.geometry.TimeDomain(0, 10)
ic1 = dde.icbc.IC(geom, lambda x: 0, boundary, component=0)
ic2 = dde.icbc.IC(geom, lambda x: 1, boundary, component=1)

if resample:
    data = dde.data.PDE(geom, ode_system, [ic1, ic2], num_train_samples_domain, 2, solution=func, num_test=100)
else:
    data = dde.data.PDE(geom, ode_system, [ic1, ic2], num_train_samples_domain + resample_times * resample_num,
                        2, solution=func, num_test=100)

plt.rcParams['font.sans-serif'] = 'Times New Roman'
plt.rcParams.update({'figure.autolayout': True})
plt.rc('font', size=18)
models = {}
if len(load) == 0:
    layer_size = [1] + [50] * 3 + [2]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.nn.FNN(layer_size, activation, initializer)

    model = dde.Model(data, net)
    model.compile("adam", lr=0.001, metrics=["l2 relative error"])
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
        model.compile("adam", lr=1e-3)
        models[prefix] = model
        losses_test[prefix] = np.array(loss_history.loss_test).sum(axis=1)
    plot_loss_combined(losses_test)
    test_nn(test_models=models)

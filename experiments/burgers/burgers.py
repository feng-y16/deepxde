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
    parser.add_argument("-ep", "--epochs", type=int, default=10000)
    parser.add_argument("-ntrd", "--num-train-samples-domain", type=int, default=3000)
    parser.add_argument("-rest", "--resample-times", type=int, default=100)
    parser.add_argument("-resn", "--resample-numbers", type=int, default=30)
    parser.add_argument("-r", "--resample", action="store_true", default=False)
    parser.add_argument("-l", "--load", nargs='+', default=[])
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
    num_results = len(models) + 1
    plt.figure(figsize=(12, 3 * num_results))
    gs = GridSpec(num_results, 1)
    X, y_true, t, x = gen_testdata()
    result_count = 0
    for legend, test_model in test_models.items():
        y_pred = test_model.predict(X)
        pde_pred = test_model.predict(X, operator=pde)
        print(legend)
        print("Mean residual:", np.mean(np.absolute(pde_pred)))
        print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))
        plt.subplot(gs[result_count, 0])
        plt.pcolormesh(t * np.ones_like(x.T), np.ones_like(t) * x.T, y_pred.reshape(len(t), len(x)), cmap="rainbow")
        plt.xlabel("t")
        plt.ylabel("x")
        plt.title("u_" + legend)
        cbar = plt.colorbar(pad=0.05, aspect=10)
        cbar.mappable.set_clim(-1, 1)
        result_count += 1
    plt.subplot(gs[-1, 0])
    plt.pcolormesh(t * np.ones_like(x.T), np.ones_like(t) * x.T, y_true.reshape(len(t), len(x)), cmap="rainbow")
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("u_exact")
    cbar = plt.colorbar(pad=0.05, aspect=10)
    cbar.mappable.set_clim(-1, 1)
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
if resample:
    prefix = "LWIS"
else:
    prefix = "PINN"
print("resample:", resample)
print("total data points:", num_train_samples_domain + resample_times * resample_num)

geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 0.99)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
ic = dde.icbc.IC(
    geomtime, lambda x: -np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial
)

if resample:
    data = dde.data.TimePDE(
        geomtime, pde, [bc, ic], num_domain=num_train_samples_domain, num_boundary=80, num_initial=160
    )
else:
    data = dde.data.TimePDE(
        geomtime, pde, [bc, ic], num_domain=num_train_samples_domain + resample_times * resample_num,
        num_boundary=80, num_initial=160
    )

plt.rcParams['font.sans-serif'] = 'Times New Roman'
plt.rcParams.update({'figure.autolayout': True})
plt.rc('font', size=18)
models = {}
if len(load) == 0:
    net = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")
    model = dde.Model(data, net)

    model.compile("adam", lr=1e-3)
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

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
    parser.add_argument("-ep", "--epochs", type=int, default=15000)
    parser.add_argument("-ntrd", "--num-train-samples-domain", type=int, default=1000)
    parser.add_argument("-rest", "--resample-times", type=int, default=100)
    parser.add_argument("-resn", "--resample-numbers", type=int, default=20)
    parser.add_argument("-r", "--resample", action="store_true", default=False)
    parser.add_argument("-l", "--load", action="store_true", default=False)
    return parser.parse_known_args()[0]


def gen_testdata():
    data = np.load(os.path.join(save_dir, "Burgers.npz"))
    t, x, exact = data["t"], data["x"], data["usol"].T
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = exact.flatten()[:, None]
    return X, y


def pde(x, y):
    dy_x = dde.grad.jacobian(y, x, i=0, j=0)
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t + y * dy_x - 0.01 / np.pi * dy_xx


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
    prefix = "ag"
else:
    prefix = "norm"
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

if not load:
    net = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")
    model = dde.Model(data, net)

    model.compile("adam", lr=1e-3)
    if resample:
        resampler = dde.callbacks.PDEGradientAccumulativeResampler(period=(epochs // (resample_times + 1) + 1) // 2,
                                                                   sample_num=resample_num, sigma=0.2)
        loss_history, train_state = model.train(epochs=epochs, callbacks=[resampler])
    else:
        loss_history, train_state = model.train(epochs=epochs)
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
    model.compile("adam", lr=1e-3)

X, y_true = gen_testdata()
y_pred = model.predict(X)
f = model.predict(X, operator=pde)
print("Mean residual:", np.mean(np.absolute(f)))
print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))

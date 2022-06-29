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
    parser.add_argument("-ntrd", "--num-train-samples-domain", type=int, default=1000)
    parser.add_argument("-rest", "--resample-times", type=int, default=4)
    parser.add_argument("-resn", "--resample-numbers", type=int, default=1000)
    parser.add_argument("-r", "--resample", action="store_true", default=False)
    parser.add_argument("-l", "--load", nargs='+', default=[])
    return parser.parse_known_args()[0]


n = 2
if dde.backend.backend_name == "pytorch":
    sin = dde.backend.pytorch.sin
else:
    from deepxde.backend import tf

    sin = tf.sin
k0 = 2 * np.pi * n


def pde(x, y):
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    dy_yy = dde.grad.hessian(y, x, i=1, j=1)

    f = k0 ** 2 * sin(k0 * x[:, 0:1]) * sin(k0 * x[:, 1:2])
    return -dy_xx - dy_yy - k0 ** 2 * y - f


def func(x):
    return np.sin(k0 * x[:, 0:1]) * np.sin(k0 * x[:, 1:2])


def transform(x, y):
    res = x[:, 0:1] * (1 - x[:, 0:1]) * x[:, 1:2] * (1 - x[:, 1:2])
    return res * y


def boundary(_, on_boundary):
    return on_boundary


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

# General parameters

precision_train = 10
precision_test = 30
hard_constraint = True
weights = 100  # if hard_constraint == False
iterations = 5000
parameters = [1e-3, 3, 150, "sin"]
learning_rate, num_dense_layers, num_dense_nodes, activation = parameters

geom = dde.geometry.Rectangle([0, 0], [1, 1])
if hard_constraint:
    bc = []
else:
    bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary)

data = dde.data.PDE(
    geom,
    pde,
    bc,
    num_domain=num_train_samples_domain,
    num_boundary=100,
    solution=func,
    num_test=10000,
)

plt.rcParams["font.sans-serif"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams.update({"figure.autolayout": True})
plt.rc("font", size=18)
models = {}

if len(load) == 0:
    net = dde.nn.FNN(
        [2] + [num_dense_nodes] * num_dense_layers + [1], activation, "Glorot uniform"
    )

    if hard_constraint:
        net.apply_output_transform(transform)

    model = dde.Model(data, net)

    if hard_constraint:
        model.compile("adam", lr=learning_rate, metrics=["l2 relative error"])
    else:
        loss_weights = [1, weights]
        model.compile(
            "adam",
            lr=learning_rate,
            metrics=["l2 relative error"],
            loss_weights=loss_weights,
        )
    losshistory, train_state = model.train(iterations=iterations)
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

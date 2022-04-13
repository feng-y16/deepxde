"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
import deepxde as dde
import numpy as np
import os
import argparse
import warnings
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ep", "--epochs", type=int, default=5000)
    parser.add_argument("-ntrd", "--num-train-samples-domain", type=int, default=35)
    parser.add_argument("-rest", "--resample-times", type=int, default=20)
    parser.add_argument("-resn", "--resample-numbers", type=int, default=1)
    parser.add_argument("-r", "--resample", action="store_true", default=False)
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


warnings.filterwarnings("ignore")
args = parse_args()
resample = args.resample
resample_times = args.resample_times
resample_num = args.resample_numbers
epochs = args.epochs
num_train_samples_domain = args.num_train_samples_domain
print("resample:", resample)
print("total data points:", num_train_samples_domain + resample_times * resample_num)

geom = dde.geometry.TimeDomain(0, 10)
ic1 = dde.icbc.IC(geom, np.sin, boundary, component=0)
ic2 = dde.icbc.IC(geom, np.cos, boundary, component=1)

if resample:
    data = dde.data.PDE(geom, ode_system, [ic1, ic2], num_train_samples_domain, 2, solution=func, num_test=100)
else:
    data = dde.data.PDE(geom, ode_system, [ic1, ic2], num_train_samples_domain + resample_times * resample_num,
                        2, solution=func, num_test=100)

layer_size = [1] + [50] * 3 + [2]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)
model.compile("adam", lr=0.001, metrics=["l2 relative error"])
if resample:
    resampler = dde.callbacks.PDEGradientAccumulativeResampler(period=epochs // (resample_times + 1) + 1,
                                                               sample_num=resample_num)
    losshistory, train_state = model.train(epochs=epochs, callbacks=[resampler])
else:
    losshistory, train_state = model.train(epochs=epochs)
# dde.saveplot(losshistory, train_state, issave=True, isplot=True)

save_dir = os.path.dirname(os.path.abspath(__file__))
if resample:
    prefix = "ag_"
else:
    prefix = ""
x_test = model.train_state.X_test
y_pred_test, _ = model._outputs_losses(False, model.train_state.X_test,
                                       model.train_state.y_test, model.train_state.test_aux_vars)
y_test = model.train_state.y_test
_, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4.5))
ax.plot(x_test, y_pred_test, label=["x1-pred", "x2-pred"])
ax.plot(x_test, y_test, label=["x1-gt", "x2-gt"])
ax.legend(loc="best")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, prefix + "result.png"))
plt.savefig(os.path.join(save_dir, prefix + "result.pdf"))
plt.close()

y_test = model.train_state.y_test
_, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4.5))
ax.plot(x_test, (y_pred_test - y_test) / np.where(np.abs(y_test) > 0, np.abs(y_test), 1.0),
        label=["x1-error", "x2-error"])
ax.legend(loc="best")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, prefix + "error.png"))
plt.savefig(os.path.join(save_dir, prefix + "error.pdf"))
plt.close()

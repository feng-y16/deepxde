from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import sciconet as scn


def main():
    def pde(x, y):
        dy_x = tf.gradients(y, x)[0]
        dy_xx = tf.gradients(dy_x, x)[0]
        return -dy_xx - np.pi ** 2 * tf.sin(np.pi * x)

    def boundary(x, on_boundary):
        return on_boundary

    def func(x):
        return np.sin(np.pi * x)

    geom = scn.geometry.Interval(-1, 1)
    bc = scn.DirichletBC(func, boundary)
    nbc = 2
    data = scn.data.PDE(geom, pde, bc, func, nbc)

    layer_size = [1] + [50] * 3 + [1]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = scn.maps.FNN(layer_size, activation, initializer)

    model = scn.Model(data, net)

    optimizer = "adam"
    lr = 0.001
    batch_size = 16
    ntest = 100
    model.compile(optimizer, lr, batch_size, ntest, metrics=["l2 relative error"])

    epochs = 10000
    losshistory, train_state = model.train(epochs)

    scn.saveplot(losshistory, train_state, issave=True, isplot=True)


if __name__ == "__main__":
    main()
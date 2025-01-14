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
from solver import solve


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--num-train-samples-domain", type=int, default=4000)
    parser.add_argument("--num-train-samples-boundary", type=int, default=200)
    parser.add_argument("--num-train-samples-initial", type=int, default=200)
    parser.add_argument("--resample-ratio", type=float, default=0.4)
    parser.add_argument("--resample-times", type=int, default=4)
    parser.add_argument("--resample-splits", type=int, default=1)
    parser.add_argument("--resample", action="store_true", default=False)
    parser.add_argument("--adversarial", action="store_true", default=False)
    parser.add_argument("--annealing", action="store_true", default=False)
    parser.add_argument("--loss-weights", nargs="+", type=float, default=[1, 1, 1, 100, 100, 100, 100])
    parser.add_argument("--load", nargs='+', default=[])
    parser.add_argument("--draw-annealing", action="store_true", default=False)
    parser.add_argument("--num-test-samples", type=int, default=100)
    parser.add_argument("--re", type=float, default=100)
    return parser.parse_known_args()[0]


a = 1
d = 1


def pde_re(re, x, u):
    u_vel, v_vel, p = u[:, 0:1], u[:, 1:2], u[:, 2:3]

    u_vel_x = dde.grad.jacobian(u, x, i=0, j=0)
    u_vel_y = dde.grad.jacobian(u, x, i=0, j=1)
    u_vel_t = dde.grad.jacobian(u, x, i=0, j=2)
    u_vel_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)
    u_vel_yy = dde.grad.hessian(u, x, component=0, i=1, j=1)

    v_vel_x = dde.grad.jacobian(u, x, i=1, j=0)
    v_vel_y = dde.grad.jacobian(u, x, i=1, j=1)
    v_vel_t = dde.grad.jacobian(u, x, i=1, j=2)
    v_vel_xx = dde.grad.hessian(u, x, component=1, i=0, j=0)
    v_vel_yy = dde.grad.hessian(u, x, component=1, i=1, j=1)

    p_x = dde.grad.jacobian(u, x, i=2, j=0)
    p_y = dde.grad.jacobian(u, x, i=2, j=1)

    momentum_x = (
            u_vel_t
            + (u_vel * u_vel_x + v_vel * u_vel_y)
            + p_x
            - 1 / re * (u_vel_xx + u_vel_yy)
    )
    momentum_y = (
            v_vel_t
            + (u_vel * v_vel_x + v_vel * v_vel_y)
            + p_y
            - 1 / re * (v_vel_xx + v_vel_yy)
    )
    continuity = u_vel_x + v_vel_y

    return [momentum_x, momentum_y, continuity]


def u_func(x):
    return np.where(x[:, 1:2] == 1, 1, 0)


def v_func(x):
    return np.zeros_like(x[:, 0:1])


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
        ax.semilogy(epochs // 20 * np.arange(len(loss)), loss, marker='o', label=legend.split("_")[0], linewidth=3)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Testing Loss")
    ax.legend(loc="best")
    plt.savefig(os.path.join(save_dir, "loss_{}.pdf".format(Re)))
    plt.savefig(os.path.join(save_dir, "loss_{}.png".format(Re)))
    plt.close()


def contour(grid, data_x, data_y, data_z, title, v_min=1, v_max=1, levels=20, resampled_points=None):
    ax = plt.subplot(grid)
    ax.contour(data_x, data_y, data_z, colors="k", linewidths=0.2, levels=levels, vmin=v_min, vmax=v_max)
    ax.contourf(data_x, data_y, data_z, cmap="rainbow", levels=levels, vmin=v_min, vmax=v_max)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    m = plt.cm.ScalarMappable(cmap="rainbow", norm=Normalize(vmin=v_min, vmax=v_max))
    m.set_array(data_z)
    m.set_clim(v_min, v_max)
    cbar = plt.colorbar(m, pad=0.03, aspect=25, format="%.0e")
    cbar.mappable.set_clim(v_min, v_max)
    if resampled_points is not None:
        # ax.scatter(resampled_points[:, 0], resampled_points[:, 1], marker='X', s=10, color='black')
        pass


def test_nn(times=None, test_models=None, draw_annealing=False):
    if test_models is None:
        test_models = {}
    if times is None:
        times = [0.5, 1.0]
    test_models_pred_exact = {}
    for legend, test_model in test_models.items():
        test_models_pred_exact[legend] = [None, None, None, None, None, None]
    exact_data_path = os.path.join(save_dir, "re_{:}.pkl".format(Re))
    if not os.path.isfile(exact_data_path):
        exact_data = solve(n_points=num_test_samples, n_iterations=10000, re=Re)
        with open(exact_data_path, "wb") as f_solver:
            pickle.dump(exact_data, f_solver)
    else:
        with open(exact_data_path, "rb") as f_solver:
            exact_data = pickle.load(f_solver)
    for time in times:
        x, y = np.meshgrid(np.linspace(0, 1, num_test_samples), np.linspace(0, 1, num_test_samples))
        X = np.vstack((np.ravel(x), np.ravel(y))).T
        t = time * np.ones(num_test_samples ** 2).reshape(num_test_samples ** 2, 1)
        X = np.hstack((X, t))
        u_exact = exact_data[time]["u"].reshape(-1)
        v_exact = exact_data[time]["v"].reshape(-1)
        p_exact = exact_data[time]["p"].reshape(-1)
        p_exact -= p_exact.mean()
        num_results = len(test_models)
        if not draw_annealing and num_results % 2 == 0:
            num_results //= 2
        num_results += 1
        plt.figure(figsize=(12, 3 * num_results))
        gs = GridSpec(num_results, 3)
        u_min = np.min(u_exact)
        u_max = np.max(u_exact)
        v_min = np.min(v_exact)
        v_max = np.max(v_exact)
        p_min = np.min(p_exact)
        p_max = np.max(p_exact)
        result_count = 0
        for legend, test_model in test_models.items():
            output = test_model.predict(X)
            u_pred = output[:, 0].reshape(-1)
            v_pred = output[:, 1].reshape(-1)
            p_pred = output[:, 2].reshape(-1)
            p_pred -= p_pred.mean()
            l2_difference_u = dde.metrics.l2_relative_error(u_exact, u_pred)
            l2_difference_v = dde.metrics.l2_relative_error(v_exact, v_pred)
            l2_difference_p = dde.metrics.l2_relative_error(p_exact, p_pred)
            if test_models_pred_exact[legend][0] is None:
                test_models_pred_exact[legend][0] = u_pred
                test_models_pred_exact[legend][1] = v_pred
                test_models_pred_exact[legend][2] = p_pred
                test_models_pred_exact[legend][3] = u_exact
                test_models_pred_exact[legend][4] = v_exact
                test_models_pred_exact[legend][5] = p_exact
            else:
                test_models_pred_exact[legend][0] = np.concatenate((test_models_pred_exact[legend][0], u_pred))
                test_models_pred_exact[legend][1] = np.concatenate((test_models_pred_exact[legend][1], v_pred))
                test_models_pred_exact[legend][2] = np.concatenate((test_models_pred_exact[legend][2], p_pred))
                test_models_pred_exact[legend][3] = np.concatenate((test_models_pred_exact[legend][3], u_exact))
                test_models_pred_exact[legend][4] = np.concatenate((test_models_pred_exact[legend][4], v_exact))
                test_models_pred_exact[legend][5] = np.concatenate((test_models_pred_exact[legend][5], p_exact))
            residual = np.mean(np.absolute(test_model.predict(X, operator=pde)))
            print(legend)
            print("Accuracy at t = {}:".format(time))
            print("Mean residual: {:.3f}".format(residual))
            print("L2 relative error in u, v, p: {:.3f} & {:.3f} & {:.3f}"
                  .format(l2_difference_u, l2_difference_v, l2_difference_p))
            top_k = 10
            error_u = np.abs(u_exact - u_pred).reshape(-1)
            error_v = np.abs(v_exact - v_pred).reshape(-1)
            error_p = np.abs(p_exact - p_pred).reshape(-1)
            error_u = error_u[np.argpartition(-error_u, top_k)[: top_k]].mean()
            error_v = error_v[np.argpartition(-error_v, top_k)[: top_k]].mean()
            error_p = error_p[np.argpartition(-error_p, top_k)[: top_k]].mean()
            print("Top {:} error in u, v, p: {:.3f} & {:.3f} & {:.3f}".format(top_k, error_u, error_v, error_p))
            if not draw_annealing and legend.split("_")[0][-2:] == "-A":
                continue
            resampled_points = test_model.resampled_data
            if resampled_points is not None:
                resampled_points = np.concatenate(resampled_points, axis=0)
                selected_index = np.where(np.abs(resampled_points[:, 2] - time) < 0.01)[0]
                resampled_points = resampled_points[selected_index][:, :2]
            contour(gs[result_count, 0], x, y, u_pred.reshape(num_test_samples, num_test_samples),
                    "u-" + legend.split("_")[0], u_min, u_max, 20, resampled_points)
            contour(gs[result_count, 1], x, y, v_pred.reshape(num_test_samples, num_test_samples),
                    "v-" + legend.split("_")[0], v_min, v_max, 20, resampled_points)
            contour(gs[result_count, 2], x, y, p_pred.reshape(num_test_samples, num_test_samples),
                    "p-" + legend.split("_")[0], p_min, p_max, 20, resampled_points)
            result_count += 1
        contour(gs[-1, 0], x, y, u_exact.reshape(num_test_samples, num_test_samples), 'u-exact', u_min, u_max, 20)
        contour(gs[-1, 1], x, y, v_exact.reshape(num_test_samples, num_test_samples), 'v-exact', v_min, v_max, 20)
        contour(gs[-1, 2], x, y, p_exact.reshape(num_test_samples, num_test_samples), 'p-exact', p_min, p_max, 20)
        plt.savefig(os.path.join(save_dir, "Re={}_t={}.png".format(Re, time)))
        plt.savefig(os.path.join(save_dir, "Re={}_t={}.pdf".format(Re, time)))
        plt.close()

    for legend, pred_true in test_models_pred_exact.items():
        u_pred, v_pred, p_pred, u_exact, v_exact, p_exact = pred_true
        l2_difference_u = dde.metrics.l2_relative_error(u_exact, u_pred)
        l2_difference_v = dde.metrics.l2_relative_error(v_exact, v_pred)
        l2_difference_p = dde.metrics.l2_relative_error(p_exact, p_pred)
        top_k = 10
        error_u = np.abs(u_exact - u_pred).reshape(-1)
        error_v = np.abs(v_exact - v_pred).reshape(-1)
        error_p = np.abs(p_exact - p_pred).reshape(-1)
        error_u = error_u[np.argpartition(-error_u, top_k)[: top_k]].mean()
        error_v = error_v[np.argpartition(-error_v, top_k)[: top_k]].mean()
        error_p = error_p[np.argpartition(-error_p, top_k)[: top_k]].mean()
        print("    {:} & {:} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f}\\\\"
              .format(legend.split("_")[1], legend.split("_")[0], l2_difference_u, l2_difference_v, l2_difference_p,
                      error_u, error_v, error_p))


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
resample_ratio = args.resample_ratio
resample_splits = args.resample_splits
epochs = args.epochs
num_train_samples_domain = args.num_train_samples_domain
num_train_samples_boundary = args.num_train_samples_boundary
num_train_samples_initial = args.num_train_samples_initial
load = args.load
num_test_samples = args.num_test_samples
Re = args.re
save_dir = os.path.dirname(os.path.abspath(__file__))
if resample:
    prefix = "LWIS"
elif adversarial:
    prefix = "AT"
else:
    prefix = "PINN"
if annealing:
    prefix += "-A"
prefix += "_{:}".format(Re)
print("resample:", resample, resample_times, resample_splits)
print("adversarial:", adversarial)
print("annealing:", annealing)
print("data points:", num_train_samples_domain, num_train_samples_boundary, num_train_samples_initial)


def pde(x, u):
    return pde_re(Re, x, u)


spatial_domain = dde.geometry.Rectangle([0, 0], [1, 1])
temporal_domain = dde.geometry.TimeDomain(0, 1)
spatio_temporal_domain = dde.geometry.GeometryXTime(spatial_domain, temporal_domain)

boundary_condition_u = dde.icbc.DirichletBC(spatio_temporal_domain,
                                            u_func, lambda _, on_boundary: on_boundary, component=0)
boundary_condition_v = dde.icbc.DirichletBC(spatio_temporal_domain,
                                            v_func, lambda _, on_boundary: on_boundary, component=1)

initial_condition_u = dde.icbc.IC(spatio_temporal_domain,
                                  u_func, lambda _, on_initial: on_initial, component=0)
initial_condition_v = dde.icbc.IC(spatio_temporal_domain,
                                  v_func, lambda _, on_initial: on_initial, component=1)

if resample or adversarial:
    data = dde.data.TimePDE(
        spatio_temporal_domain, pde,
        [boundary_condition_u, boundary_condition_v,
         initial_condition_u, initial_condition_v],
        num_domain=int(num_train_samples_domain - num_train_samples_domain * resample_ratio),
        num_boundary=int(num_train_samples_boundary - num_train_samples_boundary * resample_ratio),
        num_initial=int(num_train_samples_initial - num_train_samples_initial * resample_ratio),
        num_test=20000)
else:
    data = dde.data.TimePDE(
        spatio_temporal_domain, pde,
        [boundary_condition_u, boundary_condition_v,
         initial_condition_u, initial_condition_v],
        num_domain=num_train_samples_domain,
        num_boundary=num_train_samples_boundary,
        num_initial=num_train_samples_initial,
        num_test=20000)

plt.rcParams["font.sans-serif"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams.update({"figure.autolayout": True})
plt.rc("font", size=18)
models = {}
if len(load) == 0:
    if os.path.isfile(os.path.join(save_dir, prefix + "_info.pkl")):
        print("skipping")
        exit(0)
    net = dde.nn.FNN([3] + [50] * 5 + [3], "tanh", "Glorot normal")
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3, loss_weights=loss_weights)
    callbacks = []
    if resample:
        resampler = dde.callbacks.PDELossAccumulativeResampler(
            sample_every=(epochs // (resample_times + 1) + 1) // 3,
            sample_num_domain=int(num_train_samples_domain * resample_ratio),
            sample_num_boundary=int(num_train_samples_boundary * resample_ratio),
            sample_num_initial=int(num_train_samples_initial * resample_ratio),
            sample_times=resample_times,
            sample_splits=resample_splits)
        callbacks.append(resampler)
    elif adversarial:
        resampler = dde.callbacks.PDEAdversarialAccumulativeResampler(
            sample_every=epochs // resample_times,
            sample_num_domain=num_train_samples_domain,
            sample_num_boundary=num_train_samples_boundary,
            sample_num_initial=num_train_samples_initial,
            sample_times=resample_times,
            eta=0.01)
        callbacks.append(resampler)
    if annealing:
        resampler = dde.callbacks.PDELearningRateAnnealing(adjust_every=epochs // 20, loss_weights=loss_weights,
                                                           num_domain_losses=3)
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
    test_nn(test_models=models, draw_annealing=args.draw_annealing)
    print("draw complete", file=sys.stderr)

import sys
import time

import numpy as np
import scipy
import jax.numpy as jnp
import tensorflow as tf
from gym import spaces
import d3rlpy

from . import config
from . import gradients as grad
from . import utils
from .backend import backend_name, tf, torch, paddle
from .icbc import IC
from .utils import list_to_str, save_animation
from .gradients import jacobian


class Callback:
    """Callback base class.

    Attributes:
        model: instance of ``Model``. Reference of the model being trained.
    """

    def __init__(self):
        self.model = None

    def set_model(self, model):
        if model is not self.model:
            self.model = model
            self.init()

    def init(self):
        """Init after setting a model."""

    def on_epoch_begin(self):
        """Called at the beginning of every epoch."""

    def on_epoch_end(self):
        """Called at the end of every epoch."""

    def on_batch_begin(self):
        """Called at the beginning of every batch."""

    def on_batch_end(self):
        """Called at the end of every batch."""

    def on_train_begin(self):
        """Called at the beginning of model training."""

    def on_train_end(self):
        """Called at the end of model training."""

    def on_predict_begin(self):
        """Called at the beginning of prediction."""

    def on_predict_end(self):
        """Called at the end of prediction."""


class CallbackList(Callback):
    """Container abstracting a list of callbacks.

    Args:
        callbacks: List of ``Callback`` instances.
    """

    def __init__(self, callbacks=None):
        callbacks = callbacks or []
        self.callbacks = list(callbacks)
        self.model = None

    def set_model(self, model):
        self.model = model
        for callback in self.callbacks:
            callback.set_model(model)

    def on_epoch_begin(self):
        for callback in self.callbacks:
            callback.on_epoch_begin()

    def on_epoch_end(self):
        for callback in self.callbacks:
            callback.on_epoch_end()

    def on_batch_begin(self):
        for callback in self.callbacks:
            callback.on_batch_begin()

    def on_batch_end(self):
        for callback in self.callbacks:
            callback.on_batch_end()

    def on_train_begin(self):
        for callback in self.callbacks:
            callback.on_train_begin()

    def on_train_end(self):
        for callback in self.callbacks:
            callback.on_train_end()

    def on_predict_begin(self):
        for callback in self.callbacks:
            callback.on_predict_begin()

    def on_predict_end(self):
        for callback in self.callbacks:
            callback.on_predict_end()

    def append(self, callback):
        if not isinstance(callback, Callback):
            raise Exception(str(callback) + " is an invalid Callback object")
        self.callbacks.append(callback)


class ModelCheckpoint(Callback):
    """Save the model after every epoch.

    Args:
        filepath (string): Prefix of filenames to save the model file.
        verbose: Verbosity mode, 0 or 1.
        save_better_only: If True, only save a better model according to the quantity
            monitored. Model is only checked at validation step according to
            ``display_every`` in ``Model.train``.
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, verbose=0, save_better_only=False, period=1):
        super().__init__()
        self.filepath = filepath
        self.verbose = verbose
        self.save_better_only = save_better_only
        self.period = period

        self.monitor = "train loss"
        self.monitor_op = np.less
        self.epochs_since_last_save = 0
        self.best = np.Inf

    def on_epoch_end(self):
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save < self.period:
            return
        self.epochs_since_last_save = 0
        if self.save_better_only:
            current = self.model.train_state.best_loss_train
            if self.monitor_op(current, self.best):
                save_path = self.model.save(self.filepath, verbose=0)
                if self.verbose > 0:
                    print(
                        "Epoch {}: {} improved from {:.2e} to {:.2e}, saving model to {} ...\n".format(
                            self.model.train_state.epoch,
                            self.monitor,
                            self.best,
                            current,
                            save_path,
                        )
                    )
                self.best = current
        else:
            self.model.save(self.filepath, verbose=self.verbose)


class EarlyStopping(Callback):
    """Stop training when a monitored quantity (training or testing loss) has stopped improving.
    Only checked at validation step according to ``display_every`` in ``Model.train``.

    Args:
        min_delta: Minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: Number of epochs with no improvement
            after which training will be stopped.
        baseline: Baseline value for the monitored quantity to reach.
            Training will stop if the model doesn't show improvement
            over the baseline.
        monitor: The loss function that is monitored. Either 'loss_train' or 'loss_test'
    """

    def __init__(self, min_delta=0, patience=0, baseline=None, monitor="loss_train"):
        super().__init__()

        self.baseline = baseline
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0

        self.monitor_op = np.less
        self.min_delta *= -1

    def on_train_begin(self):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self):
        current = self.get_monitor_value()
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = self.model.train_state.epoch
                self.model.stop_training = True

    def on_train_end(self):
        if self.stopped_epoch > 0:
            print("Epoch {}: early stopping".format(self.stopped_epoch))

    def get_monitor_value(self):
        if self.monitor == "loss_train":
            result = sum(self.model.train_state.loss_train)
        elif self.monitor == "loss_test":
            result = sum(self.model.train_state.loss_test)
        else:
            raise ValueError("The specified monitor function is incorrect.")

        return result


class Timer(Callback):
    """Stop training when training time reaches the threshold.
    This Timer starts after the first call of `on_train_begin`.

    Args:
        available_time (float): Total time (in minutes) available for the training.
    """

    def __init__(self, available_time):
        super().__init__()

        self.threshold = available_time * 60  # convert to seconds
        self.t_start = None

    def on_train_begin(self):
        if self.t_start is None:
            self.t_start = time.time()

    def on_epoch_end(self):
        if time.time() - self.t_start > self.threshold:
            self.model.stop_training = True
            print(
                "\nStop training as time used up. time used: {:.1f} mins, epoch trained: {}".format(
                    (time.time() - self.t_start) / 60, self.model.train_state.epoch
                )
            )


class DropoutUncertainty(Callback):
    """Uncertainty estimation via MC dropout.

    References:
        `Y. Gal, & Z. Ghahramani. Dropout as a Bayesian approximation: Representing
        model uncertainty in deep learning. International Conference on Machine
        Learning, 2016 <https://arxiv.org/abs/1506.02142>`_.

    Warning:
        This cannot be used together with other techniques that have different behaviors
        during training and testing, such as batch normalization.
    """

    def __init__(self, period=1000):
        super().__init__()
        self.period = period
        self.epochs_since_last = 0

    def on_epoch_end(self):
        self.epochs_since_last += 1
        if self.epochs_since_last >= self.period:
            self.epochs_since_last = 0
            y_preds = []
            for _ in range(1000):
                y_pred_test_one = self.model._outputs(
                    True, self.model.train_state.X_test
                )
                y_preds.append(y_pred_test_one)
            self.model.train_state.y_std_test = np.std(y_preds, axis=0)

    def on_train_end(self):
        self.on_epoch_end()


class VariableValue(Callback):
    """Get the variable values.

    Args:
        var_list: A `TensorFlow Variable <https://www.tensorflow.org/api_docs/python/tf/Variable>`_
            or a list of TensorFlow Variable.
        period (int): Interval (number of epochs) between checking values.
        filename (string): Output the values to the file `filename`.
            The file is kept open to allow instances to be re-used.
            If ``None``, output to the screen.
        precision (int): The precision of variables to display.
    """

    def __init__(self, var_list, period=1, filename=None, precision=2):
        super().__init__()
        self.var_list = var_list if isinstance(var_list, list) else [var_list]
        self.period = period
        self.precision = precision

        self.file = sys.stdout if filename is None else open(filename, "w", buffering=1)
        self.value = None
        self.epochs_since_last = 0

    def on_train_begin(self):
        if backend_name == "tensorflow.compat.v1":
            self.value = self.model.sess.run(self.var_list)
        elif backend_name == "tensorflow":
            self.value = [var.numpy() for var in self.var_list]
        elif backend_name in ["pytorch", "paddle"]:
            self.value = [var.detach().item() for var in self.var_list]
        print(
            self.model.train_state.epoch,
            utils.list_to_str(self.value, precision=self.precision),
            file=self.file,
        )
        self.file.flush()

    def on_epoch_end(self):
        self.epochs_since_last += 1
        if self.epochs_since_last >= self.period:
            self.epochs_since_last = 0
            self.on_train_begin()

    def get_value(self):
        """Return the variable values."""
        return self.value


class OperatorPredictor(Callback):
    """Generates operator values for the input samples.

    Args:
        x: The input data.
        op: The operator with inputs (x, y).
    """

    def __init__(self, x, op):
        super().__init__()
        self.x = x
        self.op = op
        self.value = None

    def init(self):
        if backend_name == "tensorflow.compat.v1":
            self.tf_op = self.op(self.model.net.inputs, self.model.net.outputs)
        elif backend_name == "tensorflow":

            @tf.function
            def op(inputs):
                y = self.model.net(inputs)
                return self.op(inputs, y)

            self.tf_op = op
        elif backend_name == "pytorch":
            self.x = torch.as_tensor(self.x)
            self.x.requires_grad_()
        elif backend_name == "paddle":
            self.x = paddle.to_tensor(self.x, stop_gradient=False)

    def on_predict_end(self):
        if backend_name == "tensorflow.compat.v1":
            self.value = self.model.sess.run(
                self.tf_op, feed_dict=self.model.net.feed_dict(False, self.x)
            )
        elif backend_name == "tensorflow":
            self.value = utils.to_numpy(self.tf_op(self.x))
        elif backend_name == "pytorch":
            self.model.net.eval()
            outputs = self.model.net(self.x)
            self.value = utils.to_numpy(self.op(self.x, outputs))
        elif backend_name == "paddle":
            self.model.net.eval()
            outputs = self.model.net(self.x)
            self.value = utils.to_numpy(self.op(self.x, outputs))
        else:
            # TODO: other backends
            raise NotImplementedError(
                f"OperatorPredictor not implemented for backend {backend_name}."
            )

    def get_value(self):
        return self.value


class FirstDerivative(OperatorPredictor):
    """Generates the first order derivative of the outputs with respect to the inputs.

    Args:
        x: The input data.
    """

    def __init__(self, x, component_x=0, component_y=0):
        def first_derivative(x, y):
            return grad.jacobian(y, x, i=component_y, j=component_x)

        super().__init__(x, first_derivative)


class MovieDumper(Callback):
    """Dump a movie to show the training progress of the function along a line.

    Args:
        spectrum: If True, dump the spectrum of the Fourier transform.
    """

    def __init__(
            self,
            filename,
            x1,
            x2,
            num_points=100,
            period=1,
            component=0,
            save_spectrum=False,
            y_reference=None,
    ):
        super().__init__()
        self.filename = filename
        x1 = np.array(x1)
        x2 = np.array(x2)
        self.x = (
                x1 + (x2 - x1) / (num_points - 1) * np.arange(num_points)[:, None]
        ).astype(dtype=config.real(np))
        self.period = period
        self.component = component
        self.save_spectrum = save_spectrum
        self.y_reference = y_reference

        self.y = []
        self.spectrum = []
        self.epochs_since_last_save = 0

    def on_train_begin(self):
        self.y.append(self.model._outputs(False, self.x)[:, self.component])
        if self.save_spectrum:
            A = np.fft.rfft(self.y[-1])
            self.spectrum.append(np.abs(A))

    def on_epoch_end(self):
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            self.on_train_begin()

    def on_train_end(self):
        fname_x = self.filename + "_x.txt"
        fname_y = self.filename + "_y.txt"
        fname_movie = self.filename + "_y.gif"
        print(
            "\nSaving the movie of function to {}, {}, {}...".format(
                fname_x, fname_y, fname_movie
            )
        )
        np.savetxt(fname_x, self.x)
        np.savetxt(fname_y, np.array(self.y))
        if self.y_reference is None:
            utils.save_animation(fname_movie, np.ravel(self.x), self.y)
        else:
            y_reference = np.ravel(self.y_reference(self.x))
            utils.save_animation(
                fname_movie, np.ravel(self.x), self.y, y_reference=y_reference
            )

        if self.save_spectrum:
            fname_spec = self.filename + "_spectrum.txt"
            fname_movie = self.filename + "_spectrum.gif"
            print(
                "Saving the movie of spectrum to {}, {}...".format(
                    fname_spec, fname_movie
                )
            )
            np.savetxt(fname_spec, np.array(self.spectrum))
            xdata = np.arange(len(self.spectrum[0]))
            if self.y_reference is None:
                utils.save_animation(fname_movie, xdata, self.spectrum, logy=True)
            else:
                A = np.fft.rfft(y_reference)
                utils.save_animation(
                    fname_movie, xdata, self.spectrum, logy=True, y_reference=np.abs(A)
                )


class PDEResidualResampler(Callback):
    """Resample the training points for PDE losses every given period."""

    def __init__(self, period=100):
        super().__init__()
        self.period = period

        self.num_bcs_initial = None
        self.epochs_since_last_resample = 0

    def on_train_begin(self):
        self.num_bcs_initial = self.model.data.num_bcs

    def on_epoch_end(self):
        self.epochs_since_last_resample += 1
        if self.epochs_since_last_resample < self.period:
            return
        self.epochs_since_last_resample = 0
        self.model.data.resample_train_points()

        if not np.array_equal(self.num_bcs_initial, self.model.data.num_bcs):
            print("Initial value of self.num_bcs:", self.num_bcs_initial)
            print("self.model.data.num_bcs:", self.model.data.num_bcs)
            raise ValueError(
                "`num_bcs` changed! Please update the loss function by `model.compile`."
            )


class PDELearningRateAnnealing(Callback):
    """Change the weights for PDE losses every given period."""

    def __init__(self, adjust_every=100, alpha=0.1, loss_weights=None, num_domain_losses=1):
        super().__init__()
        if loss_weights is None:
            loss_weights = [1]
        self.adjust_every = adjust_every
        self.alpha = alpha
        self.loss_weights = loss_weights
        self.num_domain_losses = num_domain_losses
        self.epochs_since_last_adjust = 0
        self.num_bcs_initial = None

    def on_train_begin(self):
        self.num_bcs_initial = self.model.data.num_bcs

    def get_parameter_gradient(self, operator=None, bc=None, bc_index=None):
        if bc is None:
            assert operator is not None

            @tf.function
            def op(inputs):
                with tf.GradientTape() as tape:
                    y = self.model.net(inputs)
                    outs = operator(inputs, y)
                    loss = None
                    if type(outs) == list:
                        for out in outs:
                            if loss is None:
                                loss = tf.math.reduce_mean(out ** 2)
                            else:
                                loss += tf.math.reduce_mean(out ** 2)
                    else:
                        loss = tf.math.reduce_mean(outs ** 2)
                    trainable_variables = (
                            self.model.net.trainable_variables + self.model.external_trainable_variables
                    )
                    grads = tape.gradient(loss, trainable_variables)
                grads_size = 0
                grads_abs_sum = 0
                for grad in grads:
                    grads_size += tf.size(grad)
                    grads_abs_sum += tf.reduce_sum(tf.abs(grad))
                return grads_abs_sum / tf.cast(grads_size, dtype=tf.float32)
            return utils.to_numpy(op(self.model.data.train_x[-self.model.data.num_domain:]))
        if operator is None:
            assert bc is not None
            assert bc_index is not None

            @tf.function
            def op(inputs, boundary_gts, bc_index):
                with tf.GradientTape() as tape:
                    y = self.model.net(inputs)
                    if bc_index is None:
                        loss = tf.reduce_mean(tf.reduce_sum((y - boundary_gts) ** 2, axis=1, keepdims=True))
                    else:
                        loss = tf.reduce_mean((y[:, bc_index:bc_index+1] - boundary_gts) ** 2)
                    trainable_variables = (
                            self.model.net.trainable_variables + self.model.external_trainable_variables
                    )
                    grads = tape.gradient(loss, trainable_variables)
                grads_abs_max = 0.0
                for grad in grads:
                    grad_abs_max = tf.reduce_max(tf.abs(grad))
                    if grads_abs_max < grad_abs_max:
                        grads_abs_max = grad_abs_max
                return grads_abs_max
            train_x_bc = bc.filter(self.model.data.train_x)
            boundary_gts = bc.original_func(train_x_bc).astype(dtype=config.real(np))
            return utils.to_numpy(op(train_x_bc, boundary_gts, bc_index))

    def on_epoch_end(self):
        self.epochs_since_last_adjust += 1
        if self.epochs_since_last_adjust < self.adjust_every:
            return
        self.epochs_since_last_adjust = 0
        domain_average_grad = self.get_parameter_gradient(operator=self.model.data.pde)
        for i in range(len(self.model.data.bcs)):
            bc = self.model.data.bcs[i]
            if hasattr(bc, "component"):
                bc_index = bc.component
            else:
                bc_index = None
            boundary_max_grad = self.get_parameter_gradient(bc=bc, bc_index=bc_index)
            new_weight = boundary_max_grad / domain_average_grad
            self.loss_weights[i + self.num_domain_losses] = \
                (1 - self.alpha) * self.loss_weights[i + self.num_domain_losses] + self.alpha * new_weight
        self.model.compile("adam", lr=1e-3, loss_weights=self.loss_weights)
        print(self.loss_weights)


class PDEAdversarialAccumulativeResampler(Callback):
    """Resample the training points for PDE losses every given period."""

    def __init__(self, sample_every=100, sample_times=4, sample_num_domain=100, sample_num_boundary=0,
                 sample_num_initial=0, eta=0.01, k=10):
        super().__init__()
        self.sample_every = sample_every
        self.sample_times = sample_times
        self.sample_num_domain = sample_num_domain // sample_times
        self.sample_num_boundary = sample_num_boundary // sample_times
        self.sample_num_initial = sample_num_initial // sample_times
        self.eta = eta
        self.k = k
        self.current_sample_times = 0
        self.epochs_since_last_sample = sample_every - 1
        self.sampled_train_points = []
        self.num_bcs_initial = None

    def projected_gradient_ascent_domain(self, x, operator, geom):
        if utils.get_num_args(operator) == 2:

            @tf.function
            def op(inputs, inputs_min, inputs_max):
                y = self.model.net(inputs)
                outs = operator(inputs, y)
                loss = None
                if type(outs) == list:
                    for out in outs:
                        if loss is None:
                            loss = out ** 2
                        else:
                            loss += out ** 2
                else:
                    loss = outs ** 2
                inputs += self.eta * jacobian(loss, inputs, i=0)
                inputs = tf.where(inputs > inputs_max, inputs_max, inputs)
                inputs = tf.where(inputs < inputs_min, inputs_min, inputs)
                return inputs
        else:
            assert utils.get_num_args(operator) == 3
            aux_vars = self.model.data.auxiliary_var_fn(x).astype(config.real(np))

            @tf.function
            def op(inputs, inputs_min, inputs_max):
                y = self.model.net(inputs)
                outs = operator(inputs, y, aux_vars)
                loss = None
                if type(outs) == list:
                    for out in outs:
                        if loss is None:
                            loss = out ** 2
                        else:
                            loss += out ** 2
                else:
                    loss = outs ** 2
                inputs += self.eta * tf.sign(jacobian(loss, inputs, i=0))
                inputs = tf.where(inputs > inputs_max, inputs_max, inputs)
                inputs = tf.where(inputs < inputs_min, inputs_min, inputs)
                return inputs
        if hasattr(geom, "timedomain"):
            x_min = geom.geometry.xmin if hasattr(geom.geometry, "xmin") else np.array([geom.geometry.l])
            x_max = geom.geometry.xmax if hasattr(geom.geometry, "xmax") else np.array([geom.geometry.r])
            x_min = np.concatenate((x_min, np.array([geom.timedomain.t0])), axis=0)
            x_max = np.concatenate((x_max, np.array([geom.timedomain.t1])), axis=0)
        else:
            x_min = geom.xmin if hasattr(geom, "xmin") else np.array([geom.l])
            x_max = geom.xmax if hasattr(geom, "xmax") else np.array([geom.r])
        x_min = x_min.reshape(1, -1).repeat(x.shape[0], axis=0).astype(config.real(np))
        x_max = x_max.reshape(1, -1).repeat(x.shape[0], axis=0).astype(config.real(np))
        for _ in range(self.k):
            x = op(x, x_min, x_max)
        return utils.to_numpy(x)

    def projected_gradient_ascent_boundary(self, x, bcs, geom):

        @tf.function
        def op(inputs, inputs_min, inputs_max, boundary_gts, update_mask):
            y = self.model.net(inputs)
            loss = tf.reduce_sum((y[:, :boundary_gts.shape[-1]] - boundary_gts) ** 2, axis=1, keepdims=True)
            inputs += self.eta * update_mask * tf.sign(jacobian(loss, inputs, i=0))
            inputs = tf.where(inputs > inputs_max, inputs_max, inputs)
            inputs = tf.where(inputs < inputs_min, inputs_min, inputs)
            return inputs

        if hasattr(geom, "timedomain"):
            x_min = geom.geometry.xmin if hasattr(geom.geometry, "xmin") else np.array([geom.geometry.l])
            x_max = geom.geometry.xmax if hasattr(geom.geometry, "xmax") else np.array([geom.geometry.r])
            x_min = np.concatenate((x_min, np.array([geom.timedomain.t0])), axis=0)
            x_max = np.concatenate((x_max, np.array([geom.timedomain.t1])), axis=0)
        else:
            x_min = geom.xmin if hasattr(geom, "xmin") else np.array([geom.l])
            x_max = geom.xmax if hasattr(geom, "xmax") else np.array([geom.r])
        x_min = x_min.reshape(1, -1).repeat(x.shape[0], axis=0).astype(config.real(np))
        x_max = x_max.reshape(1, -1).repeat(x.shape[0], axis=0).astype(config.real(np))
        update_mask = np.logical_or(np.isclose(x, x_min), np.isclose(x, x_max)).astype(config.real(np))
        boundary_gts = []
        for bc in bcs:
            if type(bc) != IC or len(bcs) == 1:
                boundary_gts.append(bc.original_func(x).astype(dtype=config.real(np)))
        boundary_gts = np.concatenate(boundary_gts, axis=1).astype(dtype=config.real(np))
        for _ in range(self.k):
            x = op(x, x_min, x_max, boundary_gts, update_mask)
        return utils.to_numpy(x)

    def on_epoch_begin(self):
        self.num_bcs_initial = self.model.data.num_bcs
        self.epochs_since_last_sample += 1
        if self.epochs_since_last_sample < self.sample_every or self.current_sample_times == self.sample_times:
            return
        self.current_sample_times += 1
        self.epochs_since_last_sample = 0
        sampled_points_domain, sampled_points_boundary, sampled_points_initial = None, None, None
        if self.sample_num_domain > 0:
            random_points_domain = self.model.data.geom.random_points(self.sample_num_domain, random="pseudo")
            sampled_points_domain = self.projected_gradient_ascent_domain(random_points_domain,
                                                                          self.model.data.pde,
                                                                          self.model.data.geom)
        if self.sample_num_boundary > 0:
            random_points_boundary = self.model.data.geom.random_boundary_points(self.sample_num_boundary,
                                                                                 random="pseudo")
            sampled_points_boundary = self.projected_gradient_ascent_boundary(random_points_boundary,
                                                                              self.model.data.bcs,
                                                                              self.model.data.geom)
        if self.sample_num_initial > 0:
            random_points_initial = self.model.data.geom.random_initial_points(self.sample_num_initial,
                                                                               random="pseudo")
            sampled_points_initial = self.projected_gradient_ascent_boundary(random_points_initial,
                                                                             self.model.data.bcs,
                                                                             self.model.data.geom)
        self.model.data.replace_train_points(sampled_points_domain, sampled_points_boundary, sampled_points_initial)
        sampled_train_points = sampled_points_domain
        if self.sample_num_boundary > 0:
            sampled_train_points = np.concatenate((sampled_train_points, sampled_points_boundary), axis=0)
        if self.sample_num_initial > 0:
            sampled_train_points = np.concatenate((sampled_train_points, sampled_points_initial), axis=0)
        self.sampled_train_points.append(sampled_train_points)
        self.print_info()

    def print_info(self):
        print(self.model.data.num_domain, end=" ")
        if hasattr(self.model.data, "num_boundary"):
            print(self.model.data.num_boundary, end=" ")
        if hasattr(self.model.data, "num_initial"):
            print(self.model.data.num_initial, end=" ")
        print("")


class RLEnv:

    def __init__(self, model, state_points=100, action_points=10):
        super(RLEnv, self).__init__()
        self.model = model
        self.state_points = state_points
        self.action_points = action_points
        geom = self.model.data.geom
        if hasattr(geom, "timedomain"):
            x_min = geom.geometry.xmin if hasattr(geom.geometry, "xmin") else np.array([geom.geometry.l])
            x_max = geom.geometry.xmax if hasattr(geom.geometry, "xmax") else np.array([geom.geometry.r])
            x_min = np.concatenate((x_min, np.array([geom.timedomain.t0])), axis=0)
            x_max = np.concatenate((x_max, np.array([geom.timedomain.t1])), axis=0)
        else:
            x_min = geom.xmin if hasattr(geom, "xmin") else np.array([geom.l])
            x_max = geom.xmax if hasattr(geom, "xmax") else np.array([geom.r])
        self.observation_space = spaces.Box(np.concatenate((x_min, [0])).repeat(state_points),
                                            np.concatenate((x_max, [np.finfo(np.float32).max])).repeat(state_points))
        self.action_space = spaces.Box(x_min.repeat(action_points), x_max.repeat(action_points))
        self.dim_points = len(x_min)
        x = self.model.data.train_x[-self.state_points:]
        loss = self.get_loss_domain(x)
        self.o = np.concatenate((x, loss.reshape(-1, 1)), axis=1).reshape(-1)

    def get_loss_domain(self, x):
        operator = self.model.data.pde
        if isinstance(x, tuple):
            x = tuple(np.asarray(xi, dtype=config.real(np)) for xi in x)
        else:
            x = np.asarray(x, dtype=config.real(np))

        if utils.get_num_args(operator) == 2:

            @tf.function
            def op(inputs):
                y = self.model.net(inputs)
                outs = operator(inputs, y)
                loss = None
                if type(outs) == list:
                    for out in outs:
                        if loss is None:
                            loss = out ** 2
                        else:
                            loss += out ** 2
                else:
                    loss = outs ** 2
                return tf.reduce_sum(loss, axis=1)
        else:
            assert utils.get_num_args(operator) == 3
            aux_vars = self.model.data.auxiliary_var_fn(x).astype(config.real(np))

            @tf.function
            def op(inputs):
                y = self.model.net(inputs)
                outs = operator(inputs, y, aux_vars)
                loss = None
                if type(outs) == list:
                    for out in outs:
                        if loss is None:
                            loss = out ** 2
                        else:
                            loss += out ** 2
                else:
                    loss = outs ** 2
                return tf.reduce_sum(loss, axis=1)
        return utils.to_numpy(op(x))

    def get_loss_boundary(self, x):
        bcs = self.model.data.bcs

        @tf.function
        def op(inputs, boundary_gts):
            y = self.model.net(inputs)
            loss = tf.reduce_sum((y[:, :boundary_gts.shape[-1]] - boundary_gts) ** 2, axis=1, keepdims=True)
            return tf.reduce_sum(loss, axis=1)
        boundary_gts = []
        for bc in bcs:
            if type(bc) != IC or len(bcs) == 1:
                boundary_gts.append(bc.original_func(x))
        boundary_gts = np.concatenate(boundary_gts, axis=1).astype(dtype=config.real(np))
        return utils.to_numpy(op(x, boundary_gts))

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        x = self.model.data.train_x[-self.state_points:]
        loss = self.get_loss_domain(x)
        self.o = np.concatenate((x, loss.reshape(-1, 1)), axis=1).reshape(-1)
        return self.o

    def step(self, a):
        x = self.model.data.train_x[-self.state_points:]
        loss = self.get_loss_domain(x)
        self.o = np.concatenate((x, loss.reshape(-1, 1)), axis=1).reshape(-1)
        r = self.get_loss_domain(a.reshape(-1, self.dim_points)).mean()
        d = 0
        return self.o, r, d, {}



class PDELossAccumulativeResampler(Callback):
    """Resample the training points for PDE losses every given period."""

    def __init__(self, sample_every=100, sample_times=4, sample_num_domain=100, **kwargs):
        super().__init__()
        self.sample_every = sample_every
        self.sample_times = sample_times
        self.sample_num = sample_num_domain
        self.sample_num_domain = sample_num_domain // sample_times
        self.current_sample_times = 0
        self.epochs_since_last_sample = 0
        self.sampled_train_points = []
        self.num_bcs_initial = None
        self.boundary = False
        self.rl_env = None
        self.rl_agent = None

    def on_train_begin(self):
        self.rl_env = RLEnv(self.model, self.model.data.num_domain // 10, self.sample_num_domain)
        self.rl_agent = d3rlpy.algos.SAC(use_gpu=False)
        self.num_bcs_initial = self.model.data.num_bcs

    def get_weights_domain(self, x, operator, loss_weight=1.0):
        if isinstance(x, tuple):
            x = tuple(np.asarray(xi, dtype=config.real(np)) for xi in x)
        else:
            x = np.asarray(x, dtype=config.real(np))

        if utils.get_num_args(operator) == 2:

            @tf.function
            def op(inputs):
                y = self.model.net(inputs)
                outs = operator(inputs, y)
                loss = None
                if type(outs) == list:
                    for out in outs:
                        if loss is None:
                            loss = out ** 2
                        else:
                            loss += out ** 2
                else:
                    loss = outs ** 2
                if loss_weight == 1:
                    return tf.reduce_sum(loss, axis=1)
                return loss_weight * tf.reduce_sum(loss, axis=1) + \
                       (1 - loss_weight) * tf.reduce_sum(jacobian(loss, inputs, i=0) ** 2, axis=1)
        else:
            assert utils.get_num_args(operator) == 3
            aux_vars = self.model.data.auxiliary_var_fn(x).astype(config.real(np))

            @tf.function
            def op(inputs):
                y = self.model.net(inputs)
                outs = operator(inputs, y, aux_vars)
                loss = None
                if type(outs) == list:
                    for out in outs:
                        if loss is None:
                            loss = out ** 2
                        else:
                            loss += out ** 2
                else:
                    loss = outs ** 2
                if loss_weight == 1:
                    return tf.reduce_sum(loss, axis=1)
                return loss_weight * tf.reduce_sum(loss, axis=1) + \
                       (1 - loss_weight) * tf.reduce_sum(jacobian(loss, inputs, i=0) ** 2, axis=1)
        return utils.to_numpy(op(x))

    def get_weights_boundary(self, x, bcs, loss_weight=1.0):

        @tf.function
        def op(inputs, boundary_gts):
            y = self.model.net(inputs)
            loss = tf.reduce_sum((y[:, :boundary_gts.shape[-1]] - boundary_gts) ** 2, axis=1, keepdims=True)
            return loss_weight * tf.reduce_sum(loss, axis=1) + \
                   (1 - loss_weight) * tf.reduce_sum(jacobian(loss, inputs, i=0) ** 2, axis=1)
        boundary_gts = []
        for bc in bcs:
            if type(bc) != IC or len(bcs) == 1:
                boundary_gts.append(bc.original_func(x))
        boundary_gts = np.concatenate(boundary_gts, axis=1).astype(dtype=config.real(np))
        return utils.to_numpy(op(x, boundary_gts))

    def on_epoch_end(self):
        self.rl_agent.fit_online(self.rl_env, n_steps=1, n_steps_per_epoch=1, save_interval=1000,
                                 save_metrics=False, verbose=False, buffer=buffer)
        self.epochs_since_last_sample += 1
        if self.epochs_since_last_sample < self.sample_every or self.current_sample_times == self.sample_times:
            return
        self.current_sample_times += 1
        self.epochs_since_last_sample = 0
        # sampled_points_domain, sampled_points_boundary, sampled_points_initial = None, None, None
        sampled_points_domain = self.rl_agent.predict(self.rl_env.o.reshape(1, -1))[0]\
            .reshape(self.sample_num_domain, -1)
        self.model.data.add_train_points(None, None, boundary=False, train_x=sampled_points_domain)
        # if self.sample_num_boundary > 0:
        #     self.model.data.add_train_points(None, None, boundary=True, train_x=sampled_points_boundary)
        # if self.sample_num_initial > 0:
        #     self.model.data.add_train_points(None, None, boundary=True, initial=True, train_x=sampled_points_initial)
        sampled_train_points = sampled_points_domain
        # if self.sample_num_boundary > 0:
        #     sampled_train_points = np.concatenate((sampled_train_points, sampled_points_boundary), axis=0)
        # if self.sample_num_initial > 0:
        #     sampled_train_points = np.concatenate((sampled_train_points, sampled_points_initial), axis=0)
        self.sampled_train_points.append(sampled_train_points)
        self.print_info()

    def print_info(self):
        print(self.model.data.num_domain, end=" ")
        if hasattr(self.model.data, "num_boundary"):
            print(self.model.data.num_boundary, end=" ")
        if hasattr(self.model.data, "num_initial"):
            print(self.model.data.num_initial, end=" ")
        print("")

from functools import partial
from typing import Callable, Dict

import numpy as np

from model_compression_toolkit.data_generation.common.enums import SchedulerType


# Custom implementation of Reduce LR On Plateau schedular
# Customized for gradient taping
class CustomReduceLROnPlateau:
    def __init__(self,
                 factor: float = 0.5,  # Factor by which the learning rate will be reduced.
                 patience: int = 10,  # Number of epochs with no improvement after which learning rate will be reduced.
                 min_delta: float = 1e-4,  # Minimum change in monitored value to qualify as improvement.
                 cooldown: int = 0,  # Number of epochs to wait before resuming normal operation after reducing
                 # learning rate.
                 min_lr: float = 1e-6,  # Lower bound on the learning rate.
                 sign_number: int = 4,  # Number of significant digits to consider for comparisons.
                 optim_lr=None,  # Optimizer learning rate variable (e.g., tf.Variable).
                 ):

        self.optim_lr = optim_lr
        self.factor = factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.patience = patience
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.wait = 0
        self.best = 0
        self.monitor_op = None
        self.sign_number = sign_number
        self.reduce_lr = True
        self._reset()

    def _reset(self):
        """
        Reset the internal state of the learning rate scheduler.
        """
        self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
        self.best = np.Inf
        self.cooldown_counter = 0
        self.wait = 0

    def on_train_begin(self, logs=None):
        """
        Initialize the learning rate scheduler at the beginning of training.
        """
        self._reset()

    def on_epoch_end(self,
                     epoch: int,
                     loss: float,
                     logs=None):
        """
        Update the learning rate based on the validation loss at the end of each epoch.

        Args:
            epoch (int): Current epoch.
            loss (float): Validation loss value.
            logs (dict): Dictionary of training metrics and logs.

        Notes:
            This method should be called at the end of each epoch during training.
        """
        logs = logs or {}
        logs['lr'] = float(self.optim_lr.learning_rate.numpy())
        current = float(loss)

        if self.in_cooldown():
            self.cooldown_counter -= 1
            self.wait = 0

        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
        elif not self.in_cooldown():
            self.wait += 1
            if self.wait >= self.patience:

                old_lr = float(self.optim_lr.learning_rate.numpy())
                if old_lr > self.min_lr and self.reduce_lr:
                    new_lr = old_lr * self.factor

                    new_lr = max(new_lr, self.min_lr)
                    self.optim_lr.learning_rate.assign(new_lr)
                    self.cooldown_counter = self.cooldown
                    self.wait = 0

    def in_cooldown(self) -> bool:
        """
        Check if the learning rate scheduler is in the cooldown phase.

        Returns:
            bool: True if in cooldown, False otherwise.
        """
        return self.cooldown_counter > 0


def get_reduceonplatue_scheduler(n_iter: int, initial_lr: float):
    """
    Create a custom ReduceLROnPlateau learning rate scheduler.

    Args:
        n_iter (int): Total number of iterations.
        initial_lr (float): Initial learning rate.

    Returns:
        callable: Partial function for creating CustomReduceLROnPlateau scheduler.
    """
    return partial(CustomReduceLROnPlateau)


# Define a dictionary that maps scheduler types to functions for creating schedulers.
scheduler_step_function_dict: Dict[SchedulerType, Callable] = {
    SchedulerType.REDUCE_ON_PLATEAU: get_reduceonplatue_scheduler  # Custom ReduceLROnPlateau scheduler.
}

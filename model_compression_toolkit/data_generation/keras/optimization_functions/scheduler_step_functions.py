# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from functools import partial
from typing import Callable, Dict

import numpy as np

from model_compression_toolkit.data_generation.common.enums import SchedulerType


# Custom implementation of Reduce LR On Plateau schedular
# Customized for gradient taping
class CustomReduceLROnPlateau:
    def __init__(self,
                 factor: float = 0.5,
                 patience: int = 10,
                 min_delta: float = 1e-4,
                 cooldown: int = 0,
                 min_lr: float = 1e-6,
                 sign_number: int = 4,
                 optim_lr=None,
                 ):
        """
        Initialize a custom learning rate scheduler based on ReduceLROnPlateau.

        Args:
            factor (float): Factor by which the learning rate will be reduced.
            patience (int): Number of epochs with no improvement after which learning rate will be reduced.
            min_delta (float): Minimum change in monitored value to qualify as an improvement.
            cooldown (int): Number of epochs to wait before resuming after reducing the learning rate.
            min_lr (float): Lower bound on the learning rate.
            sign_number (int): Number of significant digits to consider for comparisons when checking for improvement.
            optim_lr (tf.Variable): Optimizer learning rate variable to synchronize with the reduced learning rate.
        """

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

    def on_epoch_end(self,
                     loss: float,
                     logs=None):
        """
        Update the learning rate based on the validation loss at the end of each epoch.

        Args:
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

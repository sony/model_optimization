# Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
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
from typing import Dict, Union

import numpy as np
import tensorflow as tf

from model_compression_toolkit.logger import Logger


class ReduceLROnPlateau(tf.keras.callbacks.Callback):
    """
    Reduce learning rate when a metric has stopped improving.
    """

    def __init__(self, optimizer: tf.keras.optimizers.Optimizer, mode: str = 'min', factor: float = 0.1,
                 patience: int = 10, threshold: float = 1e-4, threshold_mode: str = 'rel', cooldown: int = 0,
                 min_lr: float = 0, eps: float = 1e-8, verbose: bool = False):
        """
       Initialize the ReduceLROnPlateau scheduler.

       Args:
           optimizer (tf.keras.optimizers.Optimizer): Wrapped optimizer.
           mode (str): One of `min`, `max`. In `min` mode, lr will be reduced when the quantity
                       monitored has stopped decreasing; in `max` mode it will be reduced when the
                       quantity monitored has stopped increasing. Default: 'min'.
           factor (float): Factor by which the learning rate will be reduced. new_lr = lr * factor.
                           Default: 0.1.
           patience (int): Number of epochs with no improvement after which learning rate will be reduced.
                           Default: 10.
           threshold (float): Threshold for measuring the new optimum, to only focus on significant changes.
                              Default: 1e-4.
           threshold_mode (str): One of `rel`, `abs`. In `rel` mode, dynamic_threshold = best * ( 1 + threshold )
                                 in 'max' mode or best * ( 1 - threshold ) in `min` mode. In `abs` mode, dynamic_threshold
                                 = best + threshold in `max` mode or best - threshold in `min` mode. Default: 'rel'.
           cooldown (int): Number of epochs to wait before resuming normal operation after lr has been reduced.
                           Default: 0.
           min_lr (float): A lower bound on the learning rate. Default: 0.
           eps (float): Minimal decay applied to lr. If the difference between new and old lr is smaller than eps,
                        the update is ignored. Default: 1e-8.
           verbose (bool): If True, prints a message to stdout for each update. Default: False.
       """

        super(ReduceLROnPlateau, self).__init__()

        if factor >= 1.0:
            Logger.critical('Factor should be < 1.0.')  # pragma: no cover
        self.factor = factor

        self.optimizer = optimizer
        self.min_lr = min_lr
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.eps = eps
        self.last_epoch = 0
        self._init_is_better(mode=mode, threshold=threshold, threshold_mode=threshold_mode)
        self._reset()

    def _reset(self) -> None:
        """
        Resets num_bad_epochs counter and cooldown counter.
        """
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def on_epoch_end(self, epoch: int, loss: float, logs: dict = None) -> None:
        """
        Check conditions and update learning rate at the end of an epoch.

        Args:
            epoch (int): The current epoch number.
            loss (float): Validation loss value.
            logs (dict): The dictionary of logs from the epoch.
        """
        current = float(loss)

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:  # pragma: no cover
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # Ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
            self.best = self.mode_worse

    def _reduce_lr(self, epoch: int) -> None:
        """
       Reduce the learning rate for each parameter group.

       Args:
           epoch (int): The current epoch number.
       """
        old_lr = float(tf.keras.backend.get_value(self.optimizer.learning_rate))
        new_lr = max(old_lr * self.factor, self.min_lr)
        if old_lr - new_lr > self.eps:
            tf.keras.backend.set_value(self.optimizer.learning_rate, new_lr)
            if self.verbose:  # pragma: no cover
                print(f'Epoch {epoch:05d}: reducing learning rate to {new_lr:.4e}.')

    @property
    def in_cooldown(self) -> bool:
        """
        Check if the scheduler is in a cooldown period.

        Returns:
            bool: True if in cooldown period, False otherwise.
        """
        return self.cooldown_counter > 0

    def is_better(self, a: float, best: Union[float, None]) -> bool:
        """
        Determine if the new value is better than the best value based on mode and threshold.

        Args:
            a (float): The new value to compare.
            best (float): The best value to compare against.

        Returns:
            bool: True if the new value is better, False otherwise.
        """
        if best is None:
            return True

        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return a < best * rel_epsilon
        elif self.mode == 'min' and self.threshold_mode == 'abs':  # pragma: no cover
            return a < best - self.threshold
        elif self.mode == 'max' and self.threshold_mode == 'rel':  # pragma: no cover
            rel_epsilon = self.threshold + 1.
            return a > best * rel_epsilon
        else:  # mode == 'max' and threshold_mode == 'abs':
            return a > best + self.threshold  # pragma: no cover

    def _init_is_better(self, mode: str, threshold: float, threshold_mode: str) -> None:
        """
        Initialize the comparison function for determining if a new value is better.

        Args:
            mode (str): The mode for comparison, 'min' or 'max'.
            threshold (float): The threshold for comparison.
            threshold_mode (str): The mode for threshold, 'rel' or 'abs'.

        Raises:
            ValueError: If an unknown mode or threshold mode is provided.
        """
        if mode not in {'min', 'max'}:
            Logger.critical(f'mode {mode} is unknown!') # pragma: no cover
        if threshold_mode not in {'rel', 'abs'}:
            Logger.critical(f'threshold mode {threshold_mode} is unknown!') # pragma: no cover

        if mode == 'min':
            self.mode_worse = float('inf')
        else:  # mode == 'max':
            self.mode_worse = float('-inf')

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode

    def get_config(self) -> Dict:  # pragma: no cover
        """
        Return the configuration of the scheduler as a dictionary.

        Returns:
            Dict: The configuration of the scheduler.
        """
        config = {
            'factor': self.factor,
            'patience': self.patience,
            'verbose': self.verbose,
            'mode': self.mode,
            'threshold': self.threshold,
            'threshold_mode': self.threshold_mode,
            'cooldown': self.cooldown,
            'min_lr': self.min_lr,
            'eps': self.eps
        }
        base_config = super(ReduceLROnPlateau, self).get_config()
        return {**base_config, **config}

    def set_config(self, config: Dict) -> None:  # pragma: no cover
        """
        Set the configuration of the scheduler from a dictionary.

        Args:
            config (Dict): The configuration dictionary.
        """
        for key, value in config.items():
            setattr(self, key, value)


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
from torch.optim.optimizer import Optimizer
from torch import inf
from typing import Union, List, Dict, Any

from model_compression_toolkit.logger import Logger


class ReduceLROnPlateauWithReset:
    """
    Reduce learning rate when a metric has stopped improving. This scheduler allows resetting
    the learning rate to the initial value after a specified number of bad epochs.
    """

    def __init__(self, optimizer: Optimizer, mode: str = 'min', factor: float = 0.1, patience: int = 10,
                 threshold: float = 1e-4, threshold_mode: str = 'rel', cooldown: int = 0,
                 min_lr: Union[float, List[float]] = 0, eps: float = 1e-8, verbose: bool = False):
        """
        Initialize the ReduceLROnPlateauWithReset scheduler.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
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
            min_lr (float or list): A scalar or a list of scalars. A lower bound on the learning rate of all param groups
                                    or each group respectively. Default: 0.
            eps (float): Minimal decay applied to lr. If the difference between new and old lr is smaller than eps,
                         the update is ignored. Default: 1e-8.
            verbose (bool): If True, prints a message to stdout for each update. Default: False.
        """
        if factor >= 1.0:
            Logger.critical('Factor should be < 1.0.') # pragma: no cover
        self.factor = factor

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            Logger.critical('{} is not an Optimizer'.format(
                type(optimizer).__name__))  # pragma: no cover
        self.optimizer = optimizer

        if isinstance(min_lr, (list, tuple)):  # pragma: no cover
            if len(min_lr) != len(optimizer.param_groups):
                Logger.critical("expected {} min_lrs, got {}".format(
                    len(optimizer.param_groups), len(min_lr))) # pragma: no cover
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

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

        self._init_is_better()
        self._reset()

    def _reset(self) -> None:
        """
        Resets num_bad_epochs counter and cooldown counter.
        """
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics: float, epoch: Union[int, None] = None) -> None:
        """
        Update learning rate based on the given metrics.

        Args:
            metrics (float): The value of the metric to evaluate.
            epoch (int, optional): The current epoch number. If not provided, it is incremented.
        """
        # Convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        # Check if the current metrics are better than the best
        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        # Handle cooldown period
        if self.in_cooldown:  # pragma: no cover
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # Ignore any bad epochs in cooldown

        # Reduce learning rate if the number of bad epochs exceeds patience
        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
            self.best = self.mode_worse

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def _reduce_lr(self, epoch: int) -> None:
        """
        Reduce the learning rate for each parameter group.

        Args:
            epoch (int): The current epoch number.
        """
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:  # pragma: no cover
                    epoch_str = ("%.2f" if isinstance(epoch, float) else "%.5d") % epoch
                    print('Epoch {}: reducing learning rate'
                          ' of group {} to {:.4e}.'.format(epoch_str, i, new_lr))

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
        if best is None:  # pragma: no cover
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

    def _init_is_better(self) -> None:
        """
        Initialize the comparison function for determining if a new value is better.

        Raises:
            ValueError: If an unknown mode or threshold mode is provided.
        """
        if self.mode not in {'min', 'max'}:
            Logger.critical('mode ' + self.mode + ' is unknown!') # pragma: no cover
        if self.threshold_mode not in {'rel', 'abs'}:
            Logger.critical('threshold mode ' + self.threshold_mode + ' is unknown!') # pragma: no cover

        if self.mode == 'min':
            self.mode_worse = float('inf')
        else:  # mode == 'max':
            self.mode_worse = float('-inf')  # pragma: no cover

    def state_dict(self) -> Dict[str, Any]:  # pragma: no cover
        """
        Return the state of the scheduler as a dictionary.

        Returns:
            dict: The state of the scheduler.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:  # pragma: no cover
        """
        Load the scheduler state.

        Args:
            state_dict (dict): The state dictionary to load.
        """
        self.__dict__.update(state_dict)
        self._init_is_better()

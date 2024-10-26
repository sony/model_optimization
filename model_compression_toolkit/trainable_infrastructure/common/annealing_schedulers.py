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
from abc import abstractmethod


class BaseLinearAnnealingScheduler:
    def __init__(self, t_start: int, t_end: int, initial_val: float, target_val: float):
        """
        Base class for Linear annealing scheduler. Returns the corresponding annealed value per time step.

        Args:
            t_start: Time step to begin annealing.
            t_end: Time step to complete annealing.
            initial_val: Initial value before annealing.
            target_val: Target value after annealing.

        Raises:
            ValueError: If t_start is not in the range [0, t_end).
        """
        if not (0 <= t_start < t_end):
            raise ValueError(f'Expected 0 <= t_start < t_end, actual {t_end=} {t_start=}')

        self.t_start = t_start
        self.t_end = t_end
        self.initial_val = initial_val
        self.target_val = target_val

    @abstractmethod
    def _compute_factor(self, t: int) -> float:
        """
        Abstract method to compute the annealing factor based on time step `t`.

        Args:
            t: Current time step.

        Returns:
            float: Annealing factor, typically in the range [0, 1].

        Raises:
            NotImplementedError: If this method is not overridden in the subclass.
        """
        raise NotImplementedError("This method should be overridden in subclasses")

    def __call__(self, t: int) -> float:
        """
        Calculates the annealed value based on the current time step `t`.

        Args:
            t: Current time step.

        Returns:
            float: Annealed value between initial_val and target_val.
        """
        factor = self._compute_factor(t)
        return self.initial_val + factor * (self.target_val - self.initial_val)


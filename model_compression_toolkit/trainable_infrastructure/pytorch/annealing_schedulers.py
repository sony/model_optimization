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
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor


class LinearAnnealingScheduler:
    def __init__(self, t_start: int, t_end: int, initial_val: float, target_val: float):
        """
        Linear annealing scheduler. Returns the corresponding annealed value per time step.

        Args:
            t_start: time step to begin annealing.
            t_end: time step to complete annealing.
            initial_val: initial value.
            target_val: target value.
        """
        if not (0 <= t_start < t_end):
            raise ValueError(f'Expected 0 <= t_start < t_end, actual {t_end=} {t_start=}')

        self.t_start = t_start
        self.t_end = t_end
        self.initial_val = initial_val
        self.target_val = target_val

    def __call__(self, t: int) -> float:
        factor = to_torch_tensor((t - self.t_start) / (self.t_end - self.t_start)).clip(0, 1)
        return self.initial_val + factor * (self.target_val - self.initial_val)

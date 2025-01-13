# Copyright 2021 Sony Semiconductor Israel, Inc. All rights reserved.
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
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Set

import numpy as np


class RUTarget(Enum):
    """
    Resource Utilization targets for mixed-precision search.

    WEIGHTS - Weights memory.
    ACTIVATION - Activation memory.
    TOTAL - Total memory.
    BOPS - Total Bit-Operations.
    """

    WEIGHTS = 'weights'
    ACTIVATION = 'activation'
    TOTAL = 'total'
    BOPS = 'bops'


@dataclass
class ResourceUtilization:
    """
    Class to represent measurements of performance.

    weights_memory: Memory of a model's weights in bytes.
    activation_memory: Memory of a model's activation in bytes.
    total_memory: The sum of model's activation and weights memory in bytes.
    bops: The total bit-operations in the model.
    """
    weights_memory: float = np.inf
    activation_memory: float = np.inf
    total_memory: float = np.inf
    bops: float = np.inf

    def weight_restricted(self):
        return self.weights_memory < np.inf

    def activation_restricted(self):
        return self.activation_memory < np.inf

    def total_mem_restricted(self):
        return self.total_memory < np.inf

    def bops_restricted(self):
        return self.bops < np.inf

    def get_resource_utilization_dict(self) -> Dict[RUTarget, float]:
        """
        Returns: a dictionary with the ResourceUtilization object's values for each resource utilization target.
        """
        return {RUTarget.WEIGHTS: self.weights_memory,
                RUTarget.ACTIVATION: self.activation_memory,
                RUTarget.TOTAL: self.total_memory,
                RUTarget.BOPS: self.bops}

    def is_satisfied_by(self, ru: 'ResourceUtilization') -> bool:
        """
        Checks whether another ResourceUtilization object satisfies the constraints defined by the current object.

        Args:
            ru: A ResourceUtilization object to check against the current object.

        Returns:
            Whether all constraints are satisfied.
        """
        return bool(ru.weights_memory <= self.weights_memory and \
                    ru.activation_memory <= self.activation_memory and \
                    ru.total_memory <= self.total_memory and \
                    ru.bops <= self.bops)

    def get_restricted_metrics(self) -> Set[RUTarget]:
        d = self.get_resource_utilization_dict()
        return {k for k, v in d.items() if v < np.inf}

    def is_any_restricted(self) -> bool:
        return bool(self.get_restricted_metrics())

    def __repr__(self):
        return f"Weights_memory: {self.weights_memory}, " \
               f"Activation_memory: {self.activation_memory}, " \
               f"Total_memory: {self.total_memory}, " \
               f"BOPS: {self.bops}"

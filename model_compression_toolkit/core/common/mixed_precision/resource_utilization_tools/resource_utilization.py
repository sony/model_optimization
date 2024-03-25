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
from enum import Enum
from typing import Dict, Any

import numpy as np


class RUTarget(Enum):
    """
    Targets for which we define Resource Utilization metrics for mixed-precision search.
    For each target that we care to consider in a mixed-precision search, there should be defined a set of
    resource utilization computation function, resource utilization aggregation function,
    and resource utilization target (within a ResourceUtilization object).

    Whenever adding a resource utilization metric to ResourceUtilization class we should add a matching target to this enum.

    WEIGHTS - Weights memory ResourceUtilization metric.

    ACTIVATION - Activation memory ResourceUtilization metric.

    TOTAL - Total memory ResourceUtilization metric.

    BOPS - Total Bit-Operations ResourceUtilization Metric.

    """

    WEIGHTS = 'weights'
    ACTIVATION = 'activation'
    TOTAL = 'total'
    BOPS = 'bops'


class ResourceUtilization:
    """
    Class to represent measurements of performance.
    """

    def __init__(self,
                 weights_memory: float = np.inf,
                 activation_memory: float = np.inf,
                 total_memory: float = np.inf,
                 bops: float = np.inf):
        """

        Args:
            weights_memory: Memory of a model's weights in bytes. Note that this includes only coefficients that should be quantized (for example, the kernel of Conv2D in Keras will be affected by this value, while the bias will not).
            activation_memory: Memory of a model's activation in bytes, according to the given activation resource utilization metric.
            total_memory: The sum of model's activation and weights memory in bytes, according to the given total resource utilization metric.
            bops: The total bit-operations in the model.
        """
        self.weights_memory = weights_memory
        self.activation_memory = activation_memory
        self.total_memory = total_memory
        self.bops = bops

    def __repr__(self):
        return f"Weights_memory: {self.weights_memory}, " \
               f"Activation_memory: {self.activation_memory}, " \
               f"Total_memory: {self.total_memory}, " \
               f"BOPS: {self.bops}"

    def get_resource_utilization_dict(self) -> Dict[RUTarget, float]:
        """
        Returns: a dictionary with the ResourceUtilization object's values for each resource utilization target.
        """
        return {RUTarget.WEIGHTS: self.weights_memory,
                RUTarget.ACTIVATION: self.activation_memory,
                RUTarget.TOTAL: self.total_memory,
                RUTarget.BOPS: self.bops}

    def set_resource_utilization_by_target(self, ru_mapping: Dict[RUTarget, float]):
        """
        Setting a ResourceUtilization object values for each ResourceUtilization target in the given dictionary.

        Args:
            ru_mapping: A mapping from a RUTarget to a matching resource utilization value.

        """
        self.weights_memory = ru_mapping.get(RUTarget.WEIGHTS, np.inf)
        self.activation_memory = ru_mapping.get(RUTarget.ACTIVATION, np.inf)
        self.total_memory = ru_mapping.get(RUTarget.TOTAL, np.inf)
        self.bops = ru_mapping.get(RUTarget.BOPS, np.inf)

    def holds_constraints(self, ru: Any) -> bool:
        """
        Checks whether the given ResourceUtilization object holds a set of ResourceUtilization constraints defined by
        the current ResourceUtilization object.

        Args:
            ru: A ResourceUtilization object to check if it holds the constraints.

        Returns: True if all the given resource utilization values are not greater than the referenced resource utilization values.

        """
        if not isinstance(ru, ResourceUtilization):
            return False

        return ru.weights_memory <= self.weights_memory and \
               ru.activation_memory <= self.activation_memory and \
               ru.total_memory <= self.total_memory and \
               ru.bops <= self.bops

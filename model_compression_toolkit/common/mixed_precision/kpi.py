# Copyright 2021 Sony Semiconductors Israel, Inc. All rights reserved.
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

import numpy as np


class KPITarget(Enum):
    """
    Targets for which we define KPIs metrics for mixed-precision search.
    For each target that we care to consider in a mixed-precision search, there should be defined a set of
    kpi computation function, kpi aggregation function, and kpi target (within a KPI object).

    Whenever adding a kpi metric to KPI class we should add a matching target to this enum.

    WEIGHTS - KPI metric for weights quantization

    ACTIVATION - KPI metric for activation quantization

    """

    WEIGHTS = 'weights'
    ACTIVATION = 'activation'


class KPI(object):
    """
    Class to represent measurements of performance.
    """

    def __init__(self,
                 weights_memory: float = np.inf,
                 activation_memory: float = np.inf):
        """

        Args:
            weights_memory: Memory of a model's weights in bytes. Note that this includes only coefficients that should be quantized (for example, the kernel of Conv2D in Keras will be affected by this value, while the bias will not).
            activation_memory: Memory of a model's activation in bytes, according to the given activation kpi metric.
        """
        self.weights_memory = weights_memory
        self.activation_memory = activation_memory

    def __repr__(self):
        return f"Weights_memory: {self.weights_memory}, " \
               f"Activation_memory: {self.activation_memory}"

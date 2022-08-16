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
from typing import Dict

import numpy as np


class KPITarget(Enum):
    """
    Targets for which we define KPIs metrics for mixed-precision search.
    For each target that we care to consider in a mixed-precision search, there should be defined a set of
    kpi computation function, kpi aggregation function, and kpi target (within a KPI object).

    Whenever adding a kpi metric to KPI class we should add a matching target to this enum.

    WEIGHTS - Weights memory KPI metric.

    ACTIVATION - Activation memory KPI metric.

    TOTAL - Total memory KPI metric.

    BOPS - Total Bit-Operations KPI Metric.

    """

    WEIGHTS = 'weights'
    ACTIVATION = 'activation'
    TOTAL = 'total'
    BOPS = 'bops'


class KPI:
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
            activation_memory: Memory of a model's activation in bytes, according to the given activation kpi metric.
            total_memory: The sum of model's activation and weights memory in bytes, according to the given total kpi metric.
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

    def get_kpi_dict(self) -> Dict[KPITarget, float]:
        """
        Returns: a dictionary with the KPI object's values for each KPI target.
        """
        return {KPITarget.WEIGHTS: self.weights_memory,
                KPITarget.ACTIVATION: self.activation_memory,
                KPITarget.TOTAL: self.total_memory,
                KPITarget.BOPS: self.bops}

    def set_kpi_by_target(self, kpis_mapping: Dict[KPITarget, float]):
        """
        Setting a KPI object values for each KPI target in the given dictionary.

        Args:
            kpis_mapping: A mapping from a KPITarget to a matching KPI value.

        """
        self.weights_memory = kpis_mapping.get(KPITarget.WEIGHTS, np.inf)
        self.activation_memory = kpis_mapping.get(KPITarget.ACTIVATION, np.inf)
        self.total_memory = kpis_mapping.get(KPITarget.TOTAL, np.inf)
        self.bops = kpis_mapping.get(KPITarget.BOPS, np.inf)

# Copyright 2022 Sony Semiconductors Israel, Inc. All rights reserved.
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
import copy
from enum import Enum
from functools import partial
from typing import List, Any
import numpy as np

from pulp import lpSum


def sum_kpi(kpi_vector: np.ndarray) -> List[Any]:
    """
    Aggregates KPIs vector to a single KPI measure by summing all values.

    Args:
        kpi_vector: A vector with nodes' KPI values.

    Returns: A list with an lpSum object for lp problem definition with the vector's sum.

    """
    return [lpSum(kpi_vector)]


def max_kpi(kpi_vector: np.ndarray) -> List[float]:
    """
    Aggregates KPIs vector to allow max constraint in the linear programming problem formalization.
    In order to do so, we need to define a separate constraint on each value in the KPI vector,
    to be bounded by the target KPI.

    Args:
        kpi_vector: A vector with nodes' KPI values.

    Returns: A list with the vector's values, to be used to define max constraint
    in the linear programming problem formalization.

    """
    return [kpi for kpi in kpi_vector]


def total_kpi(kpi_tensor: np.ndarray) -> List[float]:
    """
    Aggregates KPIs vector to allow weights and activation total kpi constraint in the linear programming
    problem formalization. In order to do so, we need to define a separate constraint on each activation value in
    the KPI vector, combined with the sum weights kpi.
    Note that the given kpi_tensor should contain weights and activation kpi values in each entry.

    Args:
        kpi_tensor: A tensor with nodes' KPI values for weights and activation.

    Returns: A list with lpSum objects, to be used to define total constraint
    in the linear programming problem formalization.

    """
    weights_kpi = lpSum([kpi[0] for kpi in kpi_tensor])
    total_kpis = [weights_kpi + activation_kpi for _, activation_kpi in kpi_tensor]

    return total_kpis


class MpKpiAggregation(Enum):
    """
    Defines kpi aggregation functions that can be used to compute final KPI metric.
    The enum values can be used to call a function on a set of arguments.

     SUM - applies the sum_kpi function

     MAX - applies the max_kpi function

     TOTAL - applies the total_kpi function

    """
    SUM = partial(sum_kpi)
    MAX = partial(max_kpi)
    TOTAL = partial(total_kpi)

    def __call__(self, *args):
        return self.value(*args)

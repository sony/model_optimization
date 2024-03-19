# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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


def sum_ru_values(ru_vector: np.ndarray, set_constraints: bool = True) -> List[Any]:
    """
    Aggregates resource utilization vector to a single resource utilization measure by summing all values.

    Args:
        ru_vector: A vector with nodes' resource utilization values.
        set_constraints: A flag for utilizing the method for resource utilization computation of a
            given config not for LP formalization purposes.

    Returns: A list with an lpSum object for lp problem definition with the vector's sum.

    """
    if not set_constraints:
        return [0] if len(ru_vector) == 0 else [sum(ru_vector)]
    return [lpSum(ru_vector)]


def max_ru_values(ru_vector: np.ndarray, set_constraints: bool = True) -> List[float]:
    """
    Aggregates resource utilization vector to allow max constraint in the linear programming problem formalization.
    In order to do so, we need to define a separate constraint on each value in the resource utilization vector,
    to be bounded by the target resource utilization.

    Args:
        ru_vector: A vector with nodes' resource utilization values.
        set_constraints: A flag for utilizing the method for resource utilization computation of a
            given config not for LP formalization purposes.

    Returns: A list with the vector's values, to be used to define max constraint
    in the linear programming problem formalization.

    """
    if not set_constraints:
        return [0] if len(ru_vector) == 0 else [max(ru_vector)]
    return [ru for ru in ru_vector]


def total_ru(ru_tensor: np.ndarray, set_constraints: bool = True) -> List[float]:
    """
    Aggregates resource utilization vector to allow weights and activation total utilization constraint in the linear programming
    problem formalization. In order to do so, we need to define a separate constraint on each activation memory utilization value in
    the resource utilization vector, combined with the sum weights memory utilization.
    Note that the given ru_tensor should contain weights and activation utilization values in each entry.

    Args:
        ru_tensor: A tensor with nodes' resource utilization values for weights and activation.
        set_constraints: A flag for utilizing the method for resource utilization computation of a
            given config not for LP formalization purposes.

    Returns: A list with lpSum objects, to be used to define total constraint
    in the linear programming problem formalization.

    """
    if not set_constraints:
        weights_ru = sum([ru[0] for ru in ru_tensor])
        activation_ru = max([ru[1] for ru in ru_tensor])
        return [weights_ru + activation_ru]

    weights_ru = lpSum([ru[0] for ru in ru_tensor])
    total_ru = [weights_ru + activation_ru for _, activation_ru in ru_tensor]

    return total_ru


class MpRuAggregation(Enum):
    """
    Defines resource utilization aggregation functions that can be used to compute final resource utilization metric.
    The enum values can be used to call a function on a set of arguments.

     SUM - applies the sum_ru_values function

     MAX - applies the max_ru_values function

     TOTAL - applies the total_ru function

    """
    SUM = partial(sum_ru_values)
    MAX = partial(max_ru_values)
    TOTAL = partial(total_ru)

    def __call__(self, *args):
        return self.value(*args)

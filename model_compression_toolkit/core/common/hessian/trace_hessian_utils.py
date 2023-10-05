# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
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
import numpy as np
from typing import List

from model_compression_toolkit.constants import EPS, HESSIAN_OUTPUT_ALPHA


def normalize_weights(jacobians_traces: List,
                      all_outputs_indices: List[int],
                      alpha: float = HESSIAN_OUTPUT_ALPHA) -> List[float]:
    """
    Output layers or layers that come after the model's considered output layers,
    are assigned with a constant normalized value, according to the given alpha variable and the number of such
    layers.
    Other layers returned weights are normalized by dividing the jacobian-based weights value by the sum of all
    other values.

    Args:
        jacobians_traces: The approximated average jacobian-based weights of each interest point.
        all_outputs_indices: A list of indices of all nodes that consider outputs.
        alpha: A multiplication factor.

    Returns: Normalized list of jacobian-based weights (for each interest point).

    """

    sum_without_outputs = sum(
        [jacobians_traces[i] for i in range(len(jacobians_traces)) if i not in all_outputs_indices])
    normalized_grads_weights = [_get_normalized_weight(grad,
                                                       i,
                                                       sum_without_outputs,
                                                       all_outputs_indices,
                                                       alpha)
                                for i, grad in enumerate(jacobians_traces)]

    return normalized_grads_weights


def _get_normalized_weight(grad: float,
                           i: int,
                           sum_without_outputs: float,
                           all_outputs_indices: List[int],
                           alpha: float) -> float:
    """
    Normalizes the node's gradient value. If it is an output or output replacement node than the normalized value is
    a constant, otherwise, it is normalized by dividing with the sum of all gradient values.

    Args:
        grad: The gradient value.
        i: The index of the node in the sorted interest points list.
        sum_without_outputs: The sum of all gradients of nodes that are not considered outputs.
        all_outputs_indices: A list of indices of all nodes that consider outputs.
        alpha: A multiplication factor.

    Returns: A normalized jacobian-based weights.

    """

    if i in all_outputs_indices:
        return alpha / len(all_outputs_indices)
    else:
        return ((1 - alpha) * grad / (sum_without_outputs + EPS))

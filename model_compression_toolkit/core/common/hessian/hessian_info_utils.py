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
from typing import List

from model_compression_toolkit.constants import EPS, HESSIAN_OUTPUT_ALPHA


def normalize_weights(trace_hessian_approximations: List,
                      outputs_indices: List[int],
                      alpha: float = HESSIAN_OUTPUT_ALPHA) -> List[float]:
    """
    Normalize trace Hessian approximations. Output layers or layers after the model's considered output layers
    are assigned a constant normalized value. Other layers' weights are normalized by dividing the
    trace Hessian approximations value by the sum of all other values.

    Args:
        trace_hessian_approximations: Approximated average jacobian-based weights for each interest point.
        outputs_indices: Indices of all nodes considered as outputs.
        alpha: Multiplication factor.

    Returns:
            Normalized list of trace Hessian approximations for each interest point.
    """
    if len(trace_hessian_approximations)==1:
        return [1.]

    sum_without_outputs = sum(
        [trace_hessian_approximations[i] for i in range(len(trace_hessian_approximations)) if i not in outputs_indices])
    normalized_grads_weights = [_get_normalized_weight(grad,
                                                       i,
                                                       sum_without_outputs,
                                                       outputs_indices,
                                                       alpha)
                                for i, grad in enumerate(trace_hessian_approximations)]

    return normalized_grads_weights


def _get_normalized_weight(grad: float,
                           i: int,
                           sum_without_outputs: float,
                           outputs_indices: List[int],
                           alpha: float) -> float:
    """
    Normalizes the node's trace Hessian approximation value. If it is an output or output
    replacement node than the normalized value is a constant, otherwise, it is normalized
     by dividing with the sum of all trace Hessian approximations values.

    Args:
        grad: The approximation value.
        i: The index of the node in the sorted interest points list.
        sum_without_outputs: The sum of all approximations of nodes that are not considered outputs.
        outputs_indices: A list of indices of nodes that consider outputs.
        alpha: A multiplication factor.

    Returns: A normalized trace Hessian approximation.

    """

    if i in outputs_indices:
        return alpha / len(outputs_indices)
    else:
        return ((1 - alpha) * grad / (sum_without_outputs + EPS))

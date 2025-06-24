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

from collections.abc import Callable
from functools import partial

from mct_quantizers import QuantizationMethod
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.core.common.quantization.quantization_params_generation.lut_kmeans_params import \
    lut_kmeans_tensor
from model_compression_toolkit.core.common.quantization.quantization_params_generation.symmetric_selection import \
    symmetric_selection_tensor
from model_compression_toolkit.core.common.quantization.quantization_params_generation.uniform_selection import \
    uniform_selection_tensor
from model_compression_toolkit.core.common.quantization.quantization_params_generation.power_of_two_selection import \
    power_of_two_selection_tensor

weights_quant_params_fns = {
    QuantizationMethod.POWER_OF_TWO: power_of_two_selection_tensor,
    QuantizationMethod.SYMMETRIC: symmetric_selection_tensor,
    QuantizationMethod.UNIFORM: uniform_selection_tensor,
    # instantiate partial once so that equality works between containing classes
    QuantizationMethod.LUT_POT_QUANTIZER: partial(lut_kmeans_tensor, is_symmetric=False),
    QuantizationMethod.LUT_SYM_QUANTIZER: partial(lut_kmeans_tensor, is_symmetric=True)
}


def get_weights_quantization_params_fn(weights_quantization_method: QuantizationMethod) -> Callable:
    """
    Generate a function for finding weights quantization parameters.

    Args:
        weights_quantization_method: Which quantization method to use for weights.
    Returns:
        A function to find the quantization parameters.

    """
    params_fn = weights_quant_params_fns.get(weights_quantization_method)
    if not params_fn:
        Logger.critical(
            f"No parameter function found for the specified quantization method: {weights_quantization_method}")  # pragma: no cover
    return params_fn

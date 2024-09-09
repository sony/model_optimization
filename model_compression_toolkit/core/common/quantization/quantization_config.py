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

from dataclasses import dataclass, field
import math
from enum import Enum

from model_compression_toolkit.constants import MIN_THRESHOLD


class QuantizationErrorMethod(Enum):
    """
    Method for quantization threshold selection:

    NOCLIPPING - Use min/max values as thresholds.

    MSE - Use mean square error for minimizing quantization noise.

    MAE - Use mean absolute error for minimizing quantization noise.

    KL - Use KL-divergence to make signals distributions to be similar as possible.

    Lp - Use Lp-norm to minimizing quantization noise.

    HMSE - Use Hessian-based mean squared error for minimizing quantization noise. This method is using Hessian scores to factorize more valuable parameters when computing the error induced by quantization.

    """

    NOCLIPPING = 0
    MSE = 1
    MAE = 2
    KL = 4
    LP = 5
    HMSE = 6


@dataclass
class QuantizationConfig:
    """
    A class that encapsulates all the different parameters used by the library to quantize a model.

    Examples:
        You can create a quantization configuration to apply to a model. For example, to quantize a model's weights and
        activations using thresholds, with weight threshold selection based on MSE and activation threshold selection
        using NOCLIPPING (min/max), while enabling relu_bound_to_power_of_2 and weights_bias_correction,
        you can instantiate a quantization configuration like this:

        >>> import model_compression_toolkit as mct
        >>> qc = mct.core.QuantizationConfig(activation_error_method=mct.core.QuantizationErrorMethod.NOCLIPPING, weights_error_method=mct.core.QuantizationErrorMethod.MSE, relu_bound_to_power_of_2=True, weights_bias_correction=True)


        The QuantizationConfig instance can then be used in the quantization workflow,
        such as with Keras in the function: :func:~model_compression_toolkit.ptq.keras_post_training_quantization`.

    """

    activation_error_method: QuantizationErrorMethod = QuantizationErrorMethod.MSE
    weights_error_method: QuantizationErrorMethod = QuantizationErrorMethod.MSE
    relu_bound_to_power_of_2: bool = False
    weights_bias_correction: bool = True
    weights_second_moment_correction: bool = False
    input_scaling: bool = False
    softmax_shift: bool = False
    shift_negative_activation_correction: bool = True
    activation_channel_equalization: bool = False
    z_threshold: float = math.inf
    min_threshold: float = MIN_THRESHOLD
    l_p_value: int = 2
    linear_collapsing: bool = True
    residual_collapsing: bool = True
    shift_negative_ratio: float = 0.05
    shift_negative_threshold_recalculation: bool = False
    shift_negative_params_search: bool = False
    concat_threshold_update: bool = False


# Default quantization configuration the library use.
DEFAULTCONFIG = QuantizationConfig(QuantizationErrorMethod.MSE, QuantizationErrorMethod.MSE,
                                   relu_bound_to_power_of_2=False, weights_bias_correction=True,
                                   weights_second_moment_correction=False, input_scaling=False, softmax_shift=False)

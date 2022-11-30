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


import math
from enum import Enum

from model_compression_toolkit.core.common.constants import MIN_THRESHOLD


class QuantizationErrorMethod(Enum):
    """
    Method for quantization threshold selection:

    NOCLIPPING - Use min/max values as thresholds.

    MSE - Use min square error for minimizing quantization noise.

    MAE - Use min absolute error for minimizing quantization noise.

    KL - Use KL-divergence to make signals distributions to be similar as possible.

    Lp - Use Lp-norm to minimizing quantization noise.

    """

    NOCLIPPING = 0
    MSE = 1
    MAE = 2
    KL = 4
    LP = 5


class QuantizationConfig:

    def __init__(self,
                 activation_error_method: QuantizationErrorMethod = QuantizationErrorMethod.MSE,
                 weights_error_method: QuantizationErrorMethod = QuantizationErrorMethod.MSE,
                 relu_bound_to_power_of_2: bool = False,
                 weights_bias_correction: bool = True,
                 weights_per_channel_threshold: bool = True,
                 weights_second_moment_correction: bool = False,
                 input_scaling: bool = False,
                 softmax_shift: bool = False,
                 shift_negative_activation_correction: bool = False,
                 activation_channel_equalization: bool = False,
                 z_threshold: float = math.inf,
                 min_threshold: float = MIN_THRESHOLD,
                 l_p_value: int = 2,
                 linear_collapsing: bool = True,
                 residual_collapsing: bool = True,
                 shift_negative_ratio: float = 0.05,
                 shift_negative_threshold_recalculation: bool = False,
                 shift_negative_params_search: bool = False):
        """
        Class to wrap all different parameters the library quantize the input model according to.

        Args:
            activation_error_method (QuantizationErrorMethod): Which method to use from QuantizationErrorMethod for activation quantization threshold selection.
            weights_error_method (QuantizationErrorMethod): Which method to use from QuantizationErrorMethod for activation quantization threshold selection.
            relu_bound_to_power_of_2 (bool): Whether to use relu to power of 2 scaling correction or not.
            weights_bias_correction (bool): Whether to use weights bias correction or not.
            weights_second_moment_correction (bool): Whether to use weights second_moment correction or not.
            weights_per_channel_threshold (bool): Whether to quantize the weights per-channel or not (per-tensor).
            input_scaling (bool): Whether to use input scaling or not.
            softmax_shift (bool): Whether to use softmax shift or not.
            shift_negative_activation_correction (bool): Whether to use shifting negative activation correction or not.
            activation_channel_equalization (bool): Whether to use activation channel equalization correction or not.
            z_threshold (float): Value of z score for outliers removal.
            min_threshold (float): Minimum threshold to use during thresholds selection.
            l_p_value (int): The p value of L_p norm threshold selection.
            block_collapsing (bool): Whether to collapse block one to another in the input network
            shift_negative_ratio (float): Value for the ratio between the minimal negative value of a non-linearity output to its activation threshold, which above it - shifting negative activation should occur if enabled.
            shift_negative_threshold_recalculation (bool): Whether or not to recompute the threshold after shifting negative activation.
            shift_negative_params_search (bool): Whether to search for optimal shift and threshold in shift negative activation (experimental)

        Examples:
            One may create a quantization configuration to quantize a model according to.
            For example, to quantize a model's weights and activation using thresholds, such that
            weights threshold selection is done using MSE, activation threshold selection is done using NOCLIPPING (min/max),
            enabling relu_bound_to_power_of_2, weights_bias_correction, and quantizing the weights per-channel,
            one can instantiate a quantization configuration:

            >>> import model_compression_toolkit as mct
            >>> qc = mct.QuantizationConfig(activation_error_method=mct.QuantizationErrorMethod.NOCLIPPING,weights_error_method=mct.QuantizationErrorMethod.MSE,relu_bound_to_power_of_2=True,weights_bias_correction=True,weights_per_channel_threshold=True)


            The QuantizationConfig instanse can then be passed to
            :func:`~model_compression_toolkit.ptq.keras_post_training_quantization`

        """

        self.activation_error_method = activation_error_method
        self.weights_error_method = weights_error_method
        self.relu_bound_to_power_of_2 = relu_bound_to_power_of_2
        self.weights_bias_correction = weights_bias_correction
        self.weights_second_moment_correction = weights_second_moment_correction
        self.weights_per_channel_threshold = weights_per_channel_threshold
        self.activation_channel_equalization = activation_channel_equalization
        self.input_scaling = input_scaling
        self.softmax_shift = softmax_shift
        self.min_threshold = min_threshold
        self.shift_negative_activation_correction = shift_negative_activation_correction
        self.z_threshold = z_threshold
        self.l_p_value = l_p_value
        self.linear_collapsing = linear_collapsing
        self.residual_collapsing = residual_collapsing
        self.shift_negative_ratio = shift_negative_ratio
        self.shift_negative_threshold_recalculation = shift_negative_threshold_recalculation
        self.shift_negative_params_search = shift_negative_params_search

    def __repr__(self):
        return str(self.__dict__)


# Default quantization configuration the library use.
DEFAULTCONFIG = QuantizationConfig(QuantizationErrorMethod.MSE,
                                   QuantizationErrorMethod.MSE,
                                   relu_bound_to_power_of_2=False,
                                   weights_bias_correction=True,
                                   weights_second_moment_correction=False,
                                   weights_per_channel_threshold=True,
                                   input_scaling=False,
                                   softmax_shift=False)

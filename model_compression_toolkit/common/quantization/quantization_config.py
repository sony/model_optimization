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


import math
from enum import Enum

from model_compression_toolkit.common.constants import MIN_THRESHOLD



class ThresholdSelectionMethod(Enum):
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


class QuantizationMethod(Enum):
    """
    Method for quantization function selection:

    POWER_OF_TWO - Symmetric, uniform, threshold is power of two quantization.

    KMEANS - k-means quantization.

    LUT_QUANTIZER - quantization using a look up table

    """
    POWER_OF_TWO = 0
    KMEANS = 1
    LUT_QUANTIZER = 2


class QuantizationConfig(object):

    def __init__(self,
                 activation_threshold_method: ThresholdSelectionMethod = ThresholdSelectionMethod.MSE,
                 weights_threshold_method: ThresholdSelectionMethod = ThresholdSelectionMethod.MSE,
                 activation_quantization_method: QuantizationMethod = QuantizationMethod.POWER_OF_TWO,
                 weights_quantization_method: QuantizationMethod = QuantizationMethod.POWER_OF_TWO,
                 activation_n_bits: int = 8,
                 weights_n_bits: int = 8,
                 relu_unbound_correction: bool = False,
                 weights_bias_correction: bool = True,
                 weights_per_channel_threshold: bool = True,
                 input_scaling: bool = False,
                 enable_weights_quantization: bool = True,
                 enable_activation_quantization: bool = True,
                 shift_negative_activation_correction: bool = False,
                 activation_channel_equalization: bool = False,
                 z_threshold: float = math.inf,
                 min_threshold: float = MIN_THRESHOLD,
                 l_p_value: int = 2,
                 shift_negative_ratio: float = 0.05,
                 shift_negative_threshold_recalculation: bool = False):
        """
        Class to wrap all different parameters the library quantize the input model according to.

        Args:
            activation_threshold_method (ThresholdSelectionMethod): Which method to use from ThresholdSelectionMethod for activation quantization threshold selection.
            weights_threshold_method (ThresholdSelectionMethod): Which method to use from ThresholdSelectionMethod for activation quantization threshold selection.
            activation_quantization_method (QuantizationMethod): Which method to use from QuantizationMethod for activation quantization.
            weights_quantization_method (QuantizationMethod): Which method to use from QuantizationMethod for weights quantization.
            activation_n_bits (int): Number of bits to quantize the activations.
            weights_n_bits (int): Number of bits to quantize the coefficients.
            relu_unbound_correction (bool): Whether to use relu unbound scaling correction or not.
            weights_bias_correction (bool): Whether to use weights bias correction or not.
            weights_per_channel_threshold (bool): Whether to quantize the weights per-channel or not (per-tensor).
            input_scaling (bool): Whether to use input scaling or not.
            enable_weights_quantization (bool): Whether to quantize the model weights or not.
            enable_activation_quantization (bool): Whether to quantize the model activations or not.
            shift_negative_activation_correction (bool): Whether to use shifting negative activation correction or not.
            activation_channel_equalization (bool): Whether to use activation channel equalization correction or not.
            z_threshold (float): Value of z score for outliers removal.
            min_threshold (float): Minimum threshold to use during thresholds selection.
            l_p_value (int): The p value of L_p norm threshold selection.
            shift_negative_ratio (float): Value for the ratio between the minimal negative value of a non-linearity output to its activation threshold, which above it - shifting negative activation should occur if enabled.
            shift_negative_threshold_recalculation (bool): Whether or not to recompute the threshold after shifting negative activation.

        Examples:
            One may create a quantization configuration to quantize a model according to.
            For example, to quantize a model using 6 bits for activation, 7 bits for weights,
            weights and activation quantization method is symetric uniform,
            weights threshold selection using MSE, activation threshold selection using NOCLIPPING,
            enabling relu_unbound_correction, weights_bias_correction, and quantizing the weights per-channel,
            one can instantiate a quantization configuration:

            >>> qc = QuantizationConfig(activation_n_bits=6, weights_n_bits=7, activation_quantization_method=QuantizationMethod.POWER_OF_TWO, weights_quantization_method=QuantizationMethod.POWER_OF_TWO, weights_threshold_method=ThresholdSelectionMethod.MSE, activation_threshold_method=ThresholdSelectionMethod.NOCLIPPING, relu_unbound_correction=True, weights_bias_correction=True, weights_per_channel_threshold=True)

            The QuantizationConfig instanse can then be passed to
            :func:`~model_compression_toolkit.keras_post_training_quantization`

            In order to use a different quantization method (than power-of-two that is used by default),
            one may pass a desired QuantizationMethod when instantiating a QuantizationConfig. For example:

            >>> qc = QuantizationConfig(activation_quantization_method=QuantizationMethod.LUT_QUANTIZER)

        """

        self.activation_threshold_method = activation_threshold_method
        self.weights_threshold_method = weights_threshold_method
        self.activation_quantization_method = activation_quantization_method
        self.weights_quantization_method = weights_quantization_method
        self.activation_n_bits = activation_n_bits
        self.weights_n_bits = weights_n_bits
        self.relu_unbound_correction = relu_unbound_correction
        self.weights_bias_correction = weights_bias_correction
        self.weights_per_channel_threshold = weights_per_channel_threshold
        self.enable_weights_quantization = enable_weights_quantization
        self.enable_activation_quantization = enable_activation_quantization
        self.activation_channel_equalization = activation_channel_equalization
        self.input_scaling = input_scaling
        self.min_threshold = min_threshold
        self.shift_negative_activation_correction = shift_negative_activation_correction
        self.z_threshold = z_threshold
        self.l_p_value = l_p_value
        self.shift_negative_ratio = shift_negative_ratio
        self.shift_negative_threshold_recalculation = shift_negative_threshold_recalculation

    def __repr__(self):
        return str(self.__dict__)


# Default quantization configuration the library use.
DEFAULTCONFIG = QuantizationConfig(ThresholdSelectionMethod.MSE,
                                   ThresholdSelectionMethod.MSE,
                                   QuantizationMethod.POWER_OF_TWO,
                                   QuantizationMethod.POWER_OF_TWO,
                                   8,
                                   8,
                                   False,
                                   True,
                                   True,
                                   False)





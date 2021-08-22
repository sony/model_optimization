# ===============================================================================
# Copyright (c) 2021, Sony Semiconductors Israel, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ===============================================================================


import math
from enum import Enum

from network_optimization_package.common.constants import MIN_THRESHOLD



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

    SYMMETRIC_UNIFORM - Symmetric uniform quantization.

    KMEANS - k-means quantization.

    LUT_KMEANS - k-means quantization using a look up table

    """
    SYMMETRIC_UNIFORM = 0
    KMEANS = 1
    LUT_QUANTIZER = 2


class QuantizationConfig(object):

    def __init__(self,
                 activation_threshold_method: ThresholdSelectionMethod = ThresholdSelectionMethod.MSE,
                 weights_threshold_method: ThresholdSelectionMethod = ThresholdSelectionMethod.MSE,
                 activation_quantization_method: QuantizationMethod = QuantizationMethod.SYMMETRIC_UNIFORM,
                 weights_quantization_method: QuantizationMethod = QuantizationMethod.SYMMETRIC_UNIFORM,
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
                 shift_negative_ratio: float = 0.25,
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

            >>> qc = QuantizationConfig(activation_n_bits=6, weights_n_bits=7, activation_quantization_method=QuantizationMethod.SYMMETRIC_UNIFORM, weights_quantization_method=QuantizationMethod.SYMMETRIC_UNIFORM, weights_threshold_method=ThresholdSelectionMethod.MSE, activation_threshold_method=ThresholdSelectionMethod.NOCLIPPING, relu_unbound_correction=True, weights_bias_correction=True, weights_per_channel_threshold=True)

            The QuantizationConfig instanse can then be passed to
            :func:`~network_optimization_package.keras_post_training_quantization`

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
                                   QuantizationMethod.SYMMETRIC_UNIFORM,
                                   QuantizationMethod.SYMMETRIC_UNIFORM,
                                   8,
                                   8,
                                   False,
                                   True,
                                   True,
                                   False)





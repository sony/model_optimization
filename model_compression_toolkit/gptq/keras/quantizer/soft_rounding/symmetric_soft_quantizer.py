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

import tensorflow as tf
import numpy as np

from model_compression_toolkit.gptq import RoundingType
from model_compression_toolkit.core.common import max_power_of_two
from model_compression_toolkit.target_platform_capabilities.target_platform import QuantizationMethod
from mct_quantizers import QuantizationTarget
from model_compression_toolkit.gptq.common.gptq_constants import PTQ_THRESHOLD, SCALE_PTQ, \
    SOFT_ROUNDING_GAMMA, SOFT_ROUNDING_ZETA, AUXVAR
from model_compression_toolkit.gptq.keras.quantizer import quant_utils as qutils
from typing import Dict, Any
from model_compression_toolkit.constants import THRESHOLD, MIN_THRESHOLD
from model_compression_toolkit.gptq.keras.quantizer.base_keras_gptq_quantizer import BaseKerasGPTQTrainableQuantizer
from model_compression_toolkit.gptq.keras.quantizer.quant_utils import power_of_two_max, clip, calculate_delta
from model_compression_toolkit.trainable_infrastructure import TrainableQuantizerWeightsConfig
from mct_quantizers import mark_quantizer
from model_compression_toolkit.trainable_infrastructure.common.quant_utils import \
    get_threshold_reshape_shape
from model_compression_toolkit.trainable_infrastructure.common.base_trainable_quantizer import VariableGroup


def soft_rounding_symmetric_quantizer(input_tensor: tf.Tensor,
                                      auxvar_tensor: tf.Variable,
                                      threshold_tensor: tf.Tensor,
                                      num_bits: int,
                                      signed: bool,
                                      power_of_two: bool) -> tf.Tensor:
    """
    Quantize a tensor symmetrically for GPTQ quantizers.

    Args:
        input_tensor: Tensor to quantize. values of this tensor are not changed during gptq.
        auxvar_tensor: Tensor that manifests the bit shift of the quantized weights due to gptq training.
        threshold_tensor: Tensor with values to compute the threshold.
        num_bits: Num of bits to use.
        signed: Signedness of the quantization range.
        power_of_two: Whether the threshold should be constrained or not.

    Returns:
        A quantized tensor.
    """

    if power_of_two:
        threshold_tensor = power_of_two_max(threshold_tensor)
    delta = calculate_delta(threshold_tensor, num_bits, signed)
    input_tensor = tf.stop_gradient(input_tensor)
    input_tensor_int = tf.floor(input_tensor / delta)
    tensor_q = input_tensor_int + auxvar_tensor
    min_int = -int(signed) * (2 ** (num_bits - int(signed)))
    max_int = (2 ** (num_bits - int(signed))) - 1
    return delta * clip(tensor_q, max_val=max_int, min_val=min_int)


@mark_quantizer(quantization_target=QuantizationTarget.Weights,
                quantization_method=[QuantizationMethod.POWER_OF_TWO, QuantizationMethod.SYMMETRIC],
                identifier=RoundingType.SoftQuantizer)
class SymmetricSoftRoundingGPTQ(BaseKerasGPTQTrainableQuantizer):
    """
    Trainable symmetric quantizer to optimize the rounding of the quantized values using a soft quantization method.
    """

    def __init__(self,
                 quantization_config: TrainableQuantizerWeightsConfig,
                 quantization_parameter_learning: bool = False):
        """
        Initialize a SymmetricSoftRoundingGPTQ object with parameters to use
        for the quantization.

        Args:
            quantization_config: Trainable weights quantizer config.
            quantization_parameter_learning: Whether to train the quantization threshold.
        """
        super().__init__(quantization_config)
        self.num_bits = quantization_config.weights_n_bits
        self.per_channel = quantization_config.weights_per_channel_threshold

        threshold_values = quantization_config.weights_quantization_params[THRESHOLD]
        self.threshold_shape = np.asarray(threshold_values).shape
        self.threshold_values = np.reshape(np.asarray(threshold_values), [-1]) if self.per_channel else np.asarray(
            threshold_values)

        self.quantization_axis = quantization_config.weights_channels_axis
        self.power_of_two = quantization_config.weights_quantization_method == QuantizationMethod.POWER_OF_TWO
        self.quantization_parameter_learning = quantization_parameter_learning
        self.num_channels = len(self.threshold_values) if self.per_channel else 1

        # gamma and zeta are stretch parameters for computing the rectified sigmoind function.
        # See: https://arxiv.org/pdf/2004.10568.pdf
        self.gamma = SOFT_ROUNDING_GAMMA
        self.zeta = SOFT_ROUNDING_ZETA

        self.quantizer_parameters = {}

    def initialize_quantization(self,
                                tensor_shape: Any,
                                name: str,
                                layer: Any):
        """
        Add quantizer parameters to the quantizer parameters dictionary

        Args:
            tensor_shape: tensor shape of the quantized tensor.
            name: Tensor name.
            layer: Layer to quantize.
        """

        if self.per_channel:
            reshape_shape = get_threshold_reshape_shape(tensor_shape,
                                                        quant_axis=self.quantization_axis,
                                                        quant_axis_dim=self.num_channels)
        else:
            reshape_shape = [self.num_channels]

        ptq_threshold_tensor = layer.add_weight(
            f"{name}_{PTQ_THRESHOLD}",
            shape=reshape_shape,
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=False)
        ptq_threshold_tensor.assign(self.threshold_values.reshape(reshape_shape))

        w = getattr(layer.layer, name)
        auxvar_tensor = layer.add_weight(
            f"{name}_{AUXVAR}",
            shape=list(w.shape),
            initializer=tf.keras.initializers.Constant(0.0),
            trainable=True)

        delta = qutils.calculate_delta(ptq_threshold_tensor, self.num_bits, signed=True)
        w_floor = tf.floor(w / delta)
        rest = (w / delta) - w_floor  # rest of rounding [0, 1)
        # Note that (rest - self.gamma) can't be zero since rest is positive and gamma is negative, so the division
        # is safe
        alpha = -qutils.safe_log((self.zeta - self.gamma) / (rest - self.gamma) - 1, 1e-16)  # => sigmoid(alpha) = rest

        auxvar_tensor.assign(alpha)

        # Add quantization variables
        self.add_quantizer_variable(AUXVAR, auxvar_tensor, VariableGroup.WEIGHTS)
        self.add_quantizer_variable(PTQ_THRESHOLD, ptq_threshold_tensor, VariableGroup.QPARAMS)

        if self.quantization_parameter_learning and not self.power_of_two:
            scale = layer.add_weight(
                f"{name}_{SCALE_PTQ}",
                shape=self.num_channels,
                initializer=tf.keras.initializers.Constant(1.0),
                trainable=True)
            self.add_quantizer_variable(SCALE_PTQ, scale, VariableGroup.QPARAMS)

    def get_soft_targets(self) -> tf.Tensor:
        """
        Computes the rectified sigmoid function for the quantization target parameters.

        Returns:
            A tensor with the soft rounding targets values.

        """
        return qutils.clip(
            tf.sigmoid(self.get_quantizer_variable(AUXVAR)) * (self.zeta - self.gamma) + self.gamma, 1, 0)

    def __call__(self,
                 inputs: tf.Tensor,
                 training: bool):
        """
        Quantize a tensor.

        Args:
            inputs: Input tensor to quantize.
            training: Whether the graph is in training mode.

        Returns:
            The quantized tensor.
        """

        ptq_threshold_tensor = self.get_quantizer_variable(PTQ_THRESHOLD)

        #####################################################
        # Soft Rounding
        #####################################################
        aux_var = self.get_soft_targets()
        if not training:
            aux_var = tf.cast(tf.math.greater_equal(aux_var, 0.5), tf.float32)

        if self.per_channel:
            reshape_shape = get_threshold_reshape_shape(inputs.shape,
                                                        quant_axis=self.quantization_axis,
                                                        quant_axis_dim=-1)

            ##########################################################
            # Calculate soft rounding targets and optimized threshold
            ##########################################################
            ptq_threshold_tensor_hat = tf.reshape(ptq_threshold_tensor, reshape_shape)

            #####################################################
            # Quantized Input
            #####################################################
            q_tensor = soft_rounding_symmetric_quantizer(input_tensor=inputs,
                                                         auxvar_tensor=aux_var,
                                                         threshold_tensor=ptq_threshold_tensor_hat,
                                                         num_bits=self.num_bits,
                                                         signed=True,
                                                         power_of_two=self.power_of_two)

            if self.quantization_parameter_learning and not self.power_of_two:
                scale = tf.reshape(self.get_quantizer_variable(SCALE_PTQ), reshape_shape)
                scale = tf.where(scale <= 0, MIN_THRESHOLD, scale)
                q_tensor *= scale

        else:
            q_tensor = soft_rounding_symmetric_quantizer(input_tensor=inputs,
                                                         auxvar_tensor=aux_var,
                                                         threshold_tensor=ptq_threshold_tensor.value(),
                                                         num_bits=self.num_bits,
                                                         signed=True,
                                                         power_of_two=self.power_of_two)

            if self.quantization_parameter_learning and not self.power_of_two:
                scale = self.get_quantizer_variable(SCALE_PTQ)
                scale = tf.where(scale <= 0, MIN_THRESHOLD, scale)
                q_tensor *= scale

        return q_tensor

    def get_quant_config(self) -> Dict[str, np.ndarray]:
        """
        Returns the config used to edit NodeQuantizationConfig after GPTQ retraining

        Returns:
            A dictionary of attributes the quantize_config retraining has changed during GPTQ retraining.
            Keys must match NodeQuantizationConfig attributes
        """

        if self.power_of_two:
            old_threshold = self.get_quantizer_variable(PTQ_THRESHOLD)
            old_threshold = max_power_of_two(old_threshold, MIN_THRESHOLD)

        else:
            old_threshold = self.get_quantizer_variable(PTQ_THRESHOLD)
            if self.quantization_parameter_learning:
                scale = self.get_quantizer_variable(SCALE_PTQ)
                if self.per_channel:
                    scale = tf.reshape(scale, self.threshold_shape)
                scale = tf.where(scale <= 0, MIN_THRESHOLD, scale)
                old_threshold = old_threshold * scale
            old_threshold = old_threshold.numpy()
        old_threshold = old_threshold.reshape(self.threshold_shape)
        return {THRESHOLD: old_threshold}

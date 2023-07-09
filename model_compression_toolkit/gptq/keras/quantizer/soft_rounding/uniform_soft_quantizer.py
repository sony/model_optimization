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
from model_compression_toolkit.trainable_infrastructure.common.constants import FQ_MIN, FQ_MAX
from model_compression_toolkit.target_platform_capabilities.target_platform import QuantizationMethod
from mct_quantizers import QuantizationTarget
from model_compression_toolkit.gptq.common.gptq_constants import \
    SOFT_ROUNDING_GAMMA, SOFT_ROUNDING_ZETA, AUXVAR
from model_compression_toolkit.gptq.keras.quantizer import quant_utils as qutils
from typing import Dict, Any
from model_compression_toolkit.constants import RANGE_MIN, RANGE_MAX
from model_compression_toolkit.gptq.keras.quantizer.base_keras_gptq_quantizer import BaseKerasGPTQTrainableQuantizer
from model_compression_toolkit.trainable_infrastructure import TrainableQuantizerWeightsConfig
from mct_quantizers import mark_quantizer
from model_compression_toolkit.trainable_infrastructure.common.quant_utils import \
    get_threshold_reshape_shape
from model_compression_toolkit.trainable_infrastructure.common.base_trainable_quantizer import VariableGroup


def soft_rounding_uniform_quantizer(input_tensor: tf.Tensor,
                                    auxvar_tensor: tf.Variable,
                                    min_tensor: tf.Tensor,
                                    max_tensor: tf.Tensor,
                                    num_bits: int) -> tf.Tensor:
    """
    Quantize a tensor uniformly for GPTQ quantizers.

    Args:
        input_tensor: Tensor to quantize. values of this tensor are not changed during gptq.
        auxvar_tensor: Tensor that manifests the bit shift of the quantized weights due to gptq training.
        min_tensor: Tensor with values to compute the min threshold.
        max_tensor: Tensor with values to compute the max threshold.
        num_bits: Num of bits to use.

    Returns:
        A quantized tensor.
    """
    # adjusts the quantization range so the quantization grid includes zero.
    min_range, max_range = qutils.fix_range_to_include_zero(min_tensor, max_tensor, num_bits)
    delta = qutils.calculate_delta_uniform(min_range, max_range, num_bits)
    input_tensor_int = qutils.ste_floor((input_tensor - min_range) / delta)
    tensor_q = input_tensor_int + auxvar_tensor
    return delta * qutils.ste_clip(tensor_q,
                                   min_val=0,
                                   max_val=2 ** num_bits - 1) + min_range


@mark_quantizer(quantization_target=QuantizationTarget.Weights,
                quantization_method=[QuantizationMethod.UNIFORM],
                identifier=RoundingType.SoftQuantizer)
class UniformSoftRoundingGPTQ(BaseKerasGPTQTrainableQuantizer):
    """
    Trainable uniform quantizer to optimize the rounding of the quantized values using a soft quantization method.
    """

    def __init__(self,
                 quantization_config: TrainableQuantizerWeightsConfig,
                 quantization_parameter_learning: bool = False):
        """
        Initialize a UniformSoftRoundingGPTQ object with parameters to use
        for the quantization.

        Args:
            quantization_config: Trainable weight quantizer config.
            quantization_parameter_learning: Whether to train the quantization threshold.
        """
        super().__init__(quantization_config)
        self.num_bits = quantization_config.weights_n_bits
        self.per_channel = quantization_config.weights_per_channel_threshold

        self.min_values = quantization_config.weights_quantization_params[RANGE_MIN]
        self.max_values = quantization_config.weights_quantization_params[RANGE_MAX]

        self.quantization_axis = quantization_config.weights_channels_axis
        assert quantization_parameter_learning is False, \
            "Quantization parameters learning in UniformSoftRoundingGPTQ not implemented yet"
        self.quantization_parameter_learning = quantization_parameter_learning
        self.num_channels = self.min_values.shape[self.quantization_axis] if self.per_channel else 1

        # gamma and zeta are stretch parameters for computing the rectified sigmoid function.
        # See: https://arxiv.org/pdf/2004.10568.pdf
        self.gamma = SOFT_ROUNDING_GAMMA
        self.zeta = SOFT_ROUNDING_ZETA

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

        min_tensor = layer.add_weight(
            f"{name}_{FQ_MIN}",
            shape=reshape_shape,
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=False)
        min_tensor.assign(self.min_values.reshape(reshape_shape))

        max_tensor = layer.add_weight(
            f"{name}_{FQ_MAX}",
            shape=reshape_shape,
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=False)
        max_tensor.assign(self.max_values.reshape(reshape_shape))

        w = getattr(layer.layer, name)
        auxvar_tensor = layer.add_weight(
            f"{name}_{AUXVAR}",
            shape=list(w.shape),
            initializer=tf.keras.initializers.Constant(0.0),
            trainable=True)

        w = layer.layer.depthwise_kernel if isinstance(layer.layer, (tf.keras.layers.DepthwiseConv2D,
                                                                     tf.keras.layers.DepthwiseConv1D)) \
            else layer.layer.kernel
        delta = qutils.calculate_delta_uniform(min_tensor, max_tensor, self.num_bits)
        w_clipped_normed = qutils.clip((w - min_tensor)/ delta, 0, 2 ** self.num_bits - 1)
        rest = w_clipped_normed - tf.floor(w_clipped_normed)  # rest of rounding [0, 1)
        alpha = -qutils.safe_log((self.zeta - self.gamma) / (rest - self.gamma) - 1, 1e-16)  # => sigmoid(alpha) = rest
        auxvar_tensor.assign(alpha)

        # Add quantization variables
        self.add_quantizer_variable(AUXVAR, auxvar_tensor, VariableGroup.WEIGHTS)
        self.add_quantizer_variable(RANGE_MIN, min_tensor, VariableGroup.QPARAMS)
        self.add_quantizer_variable(RANGE_MAX, max_tensor, VariableGroup.QPARAMS)

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

        min_tensor = self.get_quantizer_variable(RANGE_MIN)
        max_tensor = self.get_quantizer_variable(RANGE_MAX)

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

            #####################################################
            # Quantized Input
            #####################################################
            q_tensor = soft_rounding_uniform_quantizer(input_tensor=inputs,
                                                       auxvar_tensor=aux_var,
                                                       min_tensor=tf.reshape(min_tensor, reshape_shape),
                                                       max_tensor=tf.reshape(max_tensor, reshape_shape),
                                                       num_bits=self.num_bits)

        else:
            q_tensor = soft_rounding_uniform_quantizer(input_tensor=inputs,
                                                       auxvar_tensor=aux_var,
                                                       min_tensor=min_tensor,
                                                       max_tensor=max_tensor,
                                                       num_bits=self.num_bits)

        return q_tensor

    def get_quant_config(self) -> Dict[str, np.ndarray]:
        """
        Returns the config used to edit NodeQuantizationConfig after GPTQ retraining

        Returns:
            A dictionary of attributes the quantize_config retraining has changed during GPTQ retraining.
            Keys must match NodeQuantizationConfig attributes
        """

        return {RANGE_MIN: self.get_quantizer_variable(RANGE_MIN).numpy(),
                RANGE_MAX: self.get_quantizer_variable(RANGE_MAX).numpy()}

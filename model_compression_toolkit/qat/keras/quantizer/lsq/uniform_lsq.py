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
import tensorflow as tf
from tensorflow.python.framework.tensor_shape import TensorShape
from model_compression_toolkit.constants import RANGE_MIN, RANGE_MAX
from model_compression_toolkit.trainable_infrastructure.common.constants import FQ_MIN, FQ_MAX
from model_compression_toolkit.trainable_infrastructure import KerasTrainableQuantizationWrapper
from model_compression_toolkit.trainable_infrastructure import TrainingMethod

from mct_quantizers import mark_quantizer, QuantizationMethod, QuantizationTarget
from mct_quantizers.keras.quantizers import \
    BaseKerasInferableQuantizer, WeightsUniformInferableQuantizer, ActivationUniformInferableQuantizer

from model_compression_toolkit import constants as C

from model_compression_toolkit.qat.keras.quantizer.base_keras_qat_quantizer import BaseKerasQATTrainableQuantizer
from model_compression_toolkit.trainable_infrastructure import TrainableQuantizerWeightsConfig, \
    TrainableQuantizerActivationConfig
from model_compression_toolkit.trainable_infrastructure.common.base_trainable_quantizer import VariableGroup
from model_compression_toolkit.core.common.quantization.quantizers.quantizers_helpers import fix_range_to_include_zero
from model_compression_toolkit.qat.keras.quantizer.quant_utils import ste_round, grad_scale, adjust_range_to_include_zero


def uniform_lsq_quantizer(x: tf.Tensor,
                          min_range: tf.Tensor,
                          max_range: tf.Tensor,
                          num_bits: int,
                          min_int: int,
                          max_int:int,
                          scale_factor: float) -> tf.Tensor:
    """
    Uniform quantizer according to LSQ algorithm: https://arxiv.org/pdf/1902.08153.pdf
    Args:
        x: input to quantize
        min_range: min range of quantization values
        max_range: min range of quantization values
        num_bits: number of bits for quantization
        min_int: min clipping integer value
        max_int: max clipping integer value
        scale_factor: grad scale of LSQ algorithm
    Returns:
        A quantized tensor
    """
    min_range, max_range = adjust_range_to_include_zero(min_range, max_range, num_bits)
    delta = (max_range - min_range) / (2 ** num_bits - 1)
    delta_scaled = grad_scale(delta, scale_factor)
    rounded = ste_round((x-min_range) / delta_scaled)
    clipped = tf.math.minimum(tf.math.maximum(rounded, min_int), max_int)
    quantized = delta_scaled * clipped + min_range
    return quantized


@mark_quantizer(quantization_target=QuantizationTarget.Weights,
                quantization_method=[QuantizationMethod.UNIFORM],
                identifier=TrainingMethod.LSQ)
class LSQUniformWeightQATQuantizer(BaseKerasQATTrainableQuantizer):
    """
    Trainable constrained quantizer to quantize layer's weights.
    """

    def __init__(self, quantization_config: TrainableQuantizerWeightsConfig):
        """
        Initialize a LSQUniformWeightQATQuantizer object with parameters to use
        for the quantization.

        Args:
            quantization_config: a trainable quantizer config class with attributes for the quantization.

        """
        super().__init__(quantization_config)
        self.num_bits = self.quantization_config.weights_n_bits
        self.per_channel = self.quantization_config.weights_per_channel_threshold
        self.channel_axis = self.quantization_config.weights_channels_axis
        max_values = np.array(quantization_config.weights_quantization_params[RANGE_MAX])
        min_values = np.array(quantization_config.weights_quantization_params[RANGE_MIN])
        self.min_max_shape = np.asarray(max_values).shape
        self.max_values = np.reshape(max_values, [-1]) if self.per_channel else float(max_values)
        self.min_values = np.reshape(min_values, [-1]) if self.per_channel else float(min_values)
        self.min_int = 0
        self.max_int = 2**self.num_bits - 1
        self.scale_factor = 1.0 / np.sqrt(self.max_int * self.max_values.size)


    def initialize_quantization(self,
                                tensor_shape: TensorShape,
                                name: str,
                                layer: KerasTrainableQuantizationWrapper):
        """
        Add quantizer parameters to the quantizer parameters dictionary

        Args:
            tensor_shape: tensor shape of the quantized tensor.
            name: Tensor name.
            layer: Layer to quantize.
        """
        fq_min = layer.add_weight(
            name + FQ_MIN,
            shape=len(self.min_values) if self.per_channel else (),
            initializer=tf.keras.initializers.Constant(-1.0),
            trainable=True)
        fq_min.assign(self.min_values)

        fq_max = layer.add_weight(
            name + FQ_MAX,
            shape=len(self.max_values) if self.per_channel else (),
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=True)
        fq_max.assign(self.max_values)

        # save the quantizer added parameters for later calculations
        self.add_quantizer_variable(FQ_MIN, fq_min, VariableGroup.QPARAMS)
        self.add_quantizer_variable(FQ_MAX, fq_max, VariableGroup.QPARAMS)

    def __call__(self, inputs: tf.Tensor,
                 training: bool):
        """
        Quantize a tensor.
        Args:
            inputs: Input tensor to quantize.
            training: Whether the graph is in training mode.

        Returns:
            The quantized tensor.
        """

        min_range = tf.reshape(self.get_quantizer_variable(FQ_MIN), self.min_max_shape)
        max_range = tf.reshape(self.get_quantizer_variable(FQ_MAX), self.min_max_shape)
        q_tensor = uniform_lsq_quantizer(inputs, min_range, max_range, self.num_bits, self.min_int, self.max_int, self.scale_factor)
        return q_tensor

    def convert2inferable(self) -> BaseKerasInferableQuantizer:
        """
        Convert quantizer to inferable quantizer.

        Returns:
            BaseKerasInferableQuantizer object.
        """
        min_range, max_range = fix_range_to_include_zero(self.get_quantizer_variable(FQ_MIN).numpy(),
                                                         self.get_quantizer_variable(FQ_MAX).numpy(),
                                                         self.num_bits)
        return WeightsUniformInferableQuantizer(num_bits=self.num_bits,
                                                min_range=list(min_range.flatten()),
                                                max_range=list(max_range.flatten()),
                                                per_channel=self.per_channel,
                                                channel_axis=self.channel_axis,
                                                input_rank=len(self.min_max_shape))


@mark_quantizer(quantization_target=QuantizationTarget.Activation,
                quantization_method=[QuantizationMethod.UNIFORM],
                identifier=TrainingMethod.LSQ)
class LSQUniformActivationQATQuantizer(BaseKerasQATTrainableQuantizer):
    """
    Trainable constrained quantizer to quantize layer activations.
    """

    def __init__(self, quantization_config: TrainableQuantizerActivationConfig):
        """
        Initialize a LSQUniformActivationQATQuantizer object with parameters to use
        for the quantization.

        Args:
            quantization_config: trainable quantizer config class
        """
        super().__init__(quantization_config)

        self.num_bits = quantization_config.activation_n_bits
        self.min_range = np.array(quantization_config.activation_quantization_params[C.RANGE_MIN])
        self.max_range = np.array(quantization_config.activation_quantization_params[C.RANGE_MAX])
        self.min_int = 0
        self.max_int = 2**self.num_bits - 1

    def initialize_quantization(self,
                                tensor_shape: TensorShape,
                                name: str,
                                layer: KerasTrainableQuantizationWrapper):
        """
        Add quantizer parameters to the quantizer parameters dictionary

        Args:
            tensor_shape: tensor shape of the quantized tensor.
            name: Tensor name.
            layer: Layer to quantize.
        """
        fq_min = layer.add_weight(
            name + FQ_MIN,
            shape=(),
            initializer=tf.keras.initializers.Constant(-1.0),
            trainable=True)
        fq_min.assign(self.min_range)

        fq_max = layer.add_weight(
            name + FQ_MAX,
            shape=(),
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=True)
        fq_max.assign(self.max_range)

        # save the quantizer added parameters for later calculations
        self.add_quantizer_variable(FQ_MIN, fq_min, VariableGroup.QPARAMS)
        self.add_quantizer_variable(FQ_MAX, fq_max, VariableGroup.QPARAMS)

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

        min_range = self.get_quantizer_variable(FQ_MIN)
        max_range = self.get_quantizer_variable(FQ_MAX)
        n_channels = inputs.shape[-1]
        scale_factor = 1.0 / np.sqrt(self.max_int * n_channels)
        q_tensor = uniform_lsq_quantizer(inputs, min_range, max_range, self.num_bits, self.min_int, self.max_int, scale_factor)
        return q_tensor

    def convert2inferable(self) -> BaseKerasInferableQuantizer:
        """
        Convert quantizer to inferable quantizer.

        Returns:
            BaseKerasInferableQuantizer object.
        """
        min_range, max_range = fix_range_to_include_zero(self.get_quantizer_variable(FQ_MIN).numpy(),
                                                         self.get_quantizer_variable(FQ_MAX).numpy(),
                                                         self.num_bits)
        return ActivationUniformInferableQuantizer(num_bits=self.num_bits,
                                                   # In activation quantization is per-tensor only - thus we pass
                                                   # the min/max as lists with a len of 1
                                                   min_range=[min_range],
                                                   max_range=[max_range])

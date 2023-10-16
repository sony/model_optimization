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

from typing import Union

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.tensor_shape import TensorShape
from model_compression_toolkit.constants import SIGNED

from model_compression_toolkit.qat import TrainingMethod

from model_compression_toolkit.target_platform_capabilities.target_platform import QuantizationMethod
from model_compression_toolkit.trainable_infrastructure import KerasTrainableQuantizationWrapper
from mct_quantizers import QuantizationTarget, mark_quantizer
from model_compression_toolkit.qat.common import THRESHOLD_TENSOR
from model_compression_toolkit import constants as C

from model_compression_toolkit.qat.keras.quantizer.base_keras_qat_quantizer import BaseKerasQATTrainableQuantizer
from model_compression_toolkit.trainable_infrastructure import TrainableQuantizerWeightsConfig, \
    TrainableQuantizerActivationConfig
from mct_quantizers.keras.quantizers import WeightsPOTInferableQuantizer, WeightsSymmetricInferableQuantizer, \
    ActivationPOTInferableQuantizer, ActivationSymmetricInferableQuantizer
from model_compression_toolkit.trainable_infrastructure.common.base_trainable_quantizer import VariableGroup
from model_compression_toolkit.qat.keras.quantizer.quant_utils import ste_round, grad_scale


def symmetric_lsq_quantizer(x: tf.Tensor,
                            thresholds: tf.Tensor,
                            num_bits: int,
                            sign: bool,
                            min_int: int,
                            max_int:int,
                            scale_factor: float) -> tf.Tensor:
    """
    Symmetric quantizer according to LSQ algorithm: https://arxiv.org/pdf/1902.08153.pdf
    Args:
        x: input to quantize
        thresholds: thresholds of quantization levels
        num_bits: number of bits for quantization
        sign: whether x is signed or not
        min_int: min clipping integer value
        max_int: max clipping integer value
        scale_factor: grad scale of LSQ algorithm
    Returns:
        A quantized tensor
    """
    delta = thresholds / (2 ** (num_bits - int(sign)))
    delta_scaled = grad_scale(delta, scale_factor)
    rounded = ste_round(x / delta_scaled)
    clipped = tf.math.minimum(tf.math.maximum(rounded, min_int), max_int)
    quantized = delta_scaled * clipped
    return quantized


@mark_quantizer(quantization_target=QuantizationTarget.Weights,
                quantization_method=[QuantizationMethod.POWER_OF_TWO, QuantizationMethod.SYMMETRIC],
                identifier=TrainingMethod.LSQ)
class LSQWeightQATQuantizer(BaseKerasQATTrainableQuantizer):
    """
    Trainable constrained quantizer to quantize layer's weights.
    """

    def __init__(self, quantization_config: TrainableQuantizerWeightsConfig):
        """
        Initialize a LSQWeightQATQuantizer object with parameters to use
        for the quantization.

        Args:
            quantization_config: trainable quantizer config class
        """
        super().__init__(quantization_config)
        self.power_of_two = quantization_config.weights_quantization_method == QuantizationMethod.POWER_OF_TWO
        self.threshold_values = np.array(quantization_config.weights_quantization_params[C.THRESHOLD])
        self.threshold_shape = self.threshold_values.shape
        self.per_channel = self.quantization_config.weights_per_channel_threshold
        self.channel_axis = self.quantization_config.weights_channels_axis
        self.threshold_values = np.reshape(np.asarray(self.threshold_values), [-1]) if self.per_channel else float(self.threshold_values)
        self.num_bits = self.quantization_config.weights_n_bits
        n_pos_bits = self.num_bits - int(C.WEIGHTS_SIGNED)
        self.min_int = -int(C.WEIGHTS_SIGNED) * (2 ** n_pos_bits)
        self.max_int = 2 **n_pos_bits - 1
        self.scale_factor = 1.0 / np.sqrt(self.max_int * self.threshold_values.size)
        if self.power_of_two:
            self.threshold_values = np.power(2.0, np.ceil(np.log2(np.maximum(self.threshold_values, C.MIN_THRESHOLD))))

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
        ptq_threshold_tensor = layer.add_weight(
            name + THRESHOLD_TENSOR,
            shape=len(self.threshold_values) if self.per_channel else (),
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=True)
        ptq_threshold_tensor.assign(self.threshold_values)

        # save the quantizer added parameters for later calculations
        self.add_quantizer_variable(THRESHOLD_TENSOR, ptq_threshold_tensor, VariableGroup.QPARAMS)

    def __call__(self,
                 inputs: tf.Tensor,
                 training: bool):
        """
        Quantize a tensor.
        Args:
            inputs: Input tensor to quantize.
            training: Whether the graph is in training mode.
            weights: Dictionary of weights the quantizer can use to quantize the tensor.
            **kwargs: Additional variables the quantizer may receive.

        Returns:
            The quantized tensor.
        """

        thresholds = self.get_quantizer_variable(THRESHOLD_TENSOR)
        q_tensor = symmetric_lsq_quantizer(inputs, thresholds, self.num_bits, C.WEIGHTS_SIGNED, self.min_int, self.max_int, self.scale_factor)
        return q_tensor

    def convert2inferable(self) -> Union[WeightsPOTInferableQuantizer, WeightsSymmetricInferableQuantizer]:
        """
        Convert quantizer to inferable quantizer.

        Returns:
            BaseKerasInferableQuantizer object.
        """
        if self.power_of_two:
            thresholds = 2 ** np.ceil(np.log2(self.get_quantizer_variable(THRESHOLD_TENSOR).numpy()))
            return WeightsPOTInferableQuantizer(num_bits=self.num_bits,
                                                threshold=list(thresholds.flatten()),
                                                per_channel=self.per_channel,
                                                channel_axis=self.channel_axis,
                                                input_rank=len(self.threshold_shape))
        else:
            thresholds = self.get_quantizer_variable(THRESHOLD_TENSOR).numpy()
            return WeightsSymmetricInferableQuantizer(num_bits=self.num_bits,
                                                      threshold=list(thresholds.flatten()),
                                                      per_channel=self.per_channel,
                                                      channel_axis=self.channel_axis,
                                                      input_rank=len(self.threshold_shape))


@mark_quantizer(quantization_target=QuantizationTarget.Activation,
                quantization_method=[QuantizationMethod.POWER_OF_TWO, QuantizationMethod.SYMMETRIC],
                identifier=TrainingMethod.LSQ)
class LSQActivationQATQuantizer(BaseKerasQATTrainableQuantizer):
    """
    Trainable constrained quantizer to quantize layer activations.
    """

    def __init__(self, quantization_config: TrainableQuantizerActivationConfig):
        """
        Initialize a LSQActivationQATQuantizer object with parameters to use
        for the quantization.

        Args:
            quantization_config: trainable quantizer config class
        """
        super().__init__(quantization_config)
        self.power_of_two = quantization_config.activation_quantization_method == QuantizationMethod.POWER_OF_TWO
        self.threshold_values = float(quantization_config.activation_quantization_params[C.THRESHOLD])
        self.threshold_shape = np.asarray(self.threshold_values).shape
        self.sign = quantization_config.activation_quantization_params[SIGNED]
        self.num_bits = quantization_config.activation_n_bits
        n_pos_bits = self.num_bits - int(self.sign)
        self.min_int = -int(self.sign) * (2 ** n_pos_bits)
        self.max_int = (2 ** n_pos_bits) - 1
        if self.power_of_two:
            self.threshold_values = np.power(2.0, np.ceil(np.log2(np.maximum(self.threshold_values, C.MIN_THRESHOLD))))


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
        ptq_threshold_tensor = layer.add_weight(
            name + THRESHOLD_TENSOR,
            shape=(),
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=True)
        ptq_threshold_tensor.assign(self.threshold_values)

        # save the quantizer added parameters for later calculations
        self.add_quantizer_variable(THRESHOLD_TENSOR, ptq_threshold_tensor, VariableGroup.QPARAMS)

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

        thresholds = self.get_quantizer_variable(THRESHOLD_TENSOR)
        n_channels = inputs.shape[-1]
        scale_factor = 1.0 / np.sqrt(self.max_int * n_channels)
        q_tensor = symmetric_lsq_quantizer(inputs, thresholds, self.num_bits, self.sign, self.min_int, self.max_int, scale_factor)
        return q_tensor

    def convert2inferable(self) -> Union[ActivationPOTInferableQuantizer, ActivationSymmetricInferableQuantizer]:
        """
        Convert quantizer to inferable quantizer.

        Returns:
            BaseKerasInferableQuantizer object.
        """

        if self.power_of_two:
            thresholds = 2 ** np.ceil(np.log2(self.get_quantizer_variable(THRESHOLD_TENSOR).numpy()))
            return ActivationPOTInferableQuantizer(num_bits=self.num_bits,
                                                   # In activation quantization is per-tensor only - thus we pass
                                                   # the threshold as a list with a len of 1
                                                   threshold=[thresholds],
                                                   signed=self.sign)
        else:
            thresholds = self.get_quantizer_variable(THRESHOLD_TENSOR).numpy()
            return ActivationSymmetricInferableQuantizer(num_bits=self.num_bits,
                                                         # In activation quantization is per-tensor only - thus we
                                                         # pass the threshold as a list with a len of 1
                                                         threshold=[thresholds],
                                                         signed=self.sign)

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

from model_compression_toolkit.trainable_infrastructure import TrainingMethod

from mct_quantizers import QuantizationMethod
from model_compression_toolkit.trainable_infrastructure import KerasTrainableQuantizationWrapper
from mct_quantizers import QuantizationTarget, mark_quantizer
from model_compression_toolkit.qat.common import THRESHOLD_TENSOR
from model_compression_toolkit import constants as C

from model_compression_toolkit.qat.keras.quantizer.base_keras_qat_weight_quantizer import BaseKerasQATWeightTrainableQuantizer
from model_compression_toolkit.trainable_infrastructure import TrainableQuantizerWeightsConfig
from mct_quantizers.keras.quantizers import WeightsPOTInferableQuantizer, WeightsSymmetricInferableQuantizer
from model_compression_toolkit.trainable_infrastructure.common.base_trainable_quantizer import VariableGroup
from model_compression_toolkit.trainable_infrastructure.keras.quantizer_utils import symmetric_lsq_quantizer


@mark_quantizer(quantization_target=QuantizationTarget.Weights,
                quantization_method=[QuantizationMethod.POWER_OF_TWO, QuantizationMethod.SYMMETRIC],
                identifier=TrainingMethod.LSQ)
class LSQWeightQATQuantizer(BaseKerasQATWeightTrainableQuantizer):
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
        self.scale_factor = 1.0 / np.sqrt(self.max_int * self.threshold_values.size) if self.per_channel else 1.0 / np.sqrt(self.max_int)
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

        Returns:
            The quantized tensor.
        """

        thresholds = tf.reshape(self.get_quantizer_variable(THRESHOLD_TENSOR), self.threshold_shape)
        q_tensor = symmetric_lsq_quantizer(inputs, thresholds, self.num_bits, C.WEIGHTS_SIGNED, self.min_int,
                                           self.max_int, self.scale_factor)
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



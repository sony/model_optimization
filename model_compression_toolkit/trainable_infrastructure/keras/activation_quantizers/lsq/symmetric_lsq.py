# Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
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
from model_compression_toolkit.trainable_infrastructure.keras.quantizer_utils import symmetric_lsq_quantizer
from model_compression_toolkit.trainable_infrastructure.keras.activation_quantizers import BaseKerasActivationTrainableQuantizer
from typing import Union

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.tensor_shape import TensorShape
from model_compression_toolkit.constants import SIGNED

from model_compression_toolkit.trainable_infrastructure import TrainingMethod

from model_compression_toolkit.trainable_infrastructure import KerasTrainableQuantizationWrapper
from mct_quantizers import QuantizationTarget, mark_quantizer, QuantizationMethod
from model_compression_toolkit.qat.common import THRESHOLD_TENSOR
from model_compression_toolkit import constants as C

from model_compression_toolkit.trainable_infrastructure import TrainableQuantizerActivationConfig
from mct_quantizers.keras.quantizers import ActivationPOTInferableQuantizer, ActivationSymmetricInferableQuantizer
from model_compression_toolkit.trainable_infrastructure.common.base_trainable_quantizer import VariableGroup

@mark_quantizer(quantization_target=QuantizationTarget.Activation,
                quantization_method=[QuantizationMethod.POWER_OF_TWO, QuantizationMethod.SYMMETRIC],
                identifier=TrainingMethod.LSQ)
class LSQSymmetricActivationTrainableQuantizer(BaseKerasActivationTrainableQuantizer):
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
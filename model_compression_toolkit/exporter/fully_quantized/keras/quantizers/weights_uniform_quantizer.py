# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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
from keras.engine.base_layer import Layer
from tensorflow import TensorShape
from tensorflow_model_optimization.python.core.quantization.keras.quantizers import Quantizer

from model_compression_toolkit.core.common.quantization.quantizers.quantizers_helpers import fix_range_to_include_zero
from model_compression_toolkit.core.common.target_platform import QuantizationMethod


class WeightsUniformQuantizer(Quantizer):
    """
    Quantizer for weights quantization.
    """

    def __init__(self,
                 nbits: int,
                 min_range: np.ndarray,
                 max_range: np.ndarray,
                 weight: tf.Tensor,
                 quantization_method: QuantizationMethod):
        """

        Args:
            nbits: Number of bits to quantize.
            min_range: Min quantization range.
            max_range: Max quantization range.
            weight: Tensor of weights to quantize.
            quantization_method: Quantization method that is used (POT, Uniform, etc.)
        """

        super().__init__()

        min_range, max_range = fix_range_to_include_zero(np.array(min_range),
                                                         np.array(max_range),
                                                         nbits)
        self.nbits = nbits
        self.min_range = tf.Variable(min_range,
                                     trainable=False,
                                     dtype=tf.float32)
        self.max_range = tf.Variable(max_range,
                                     trainable=False,
                                     dtype=tf.float32)

        self.delta = (self.max_range - self.min_range) / (2 ** self.nbits - 1)
        self.weight = self._uniform_quantize(weight)
        self.quantization_method = quantization_method

    def _uniform_quantize(self, weight: tf.Tensor) -> tf.Tensor:
        """
        Quantize Tensor uniformly.

        Args:
            weight: Tensor to quantize.

        Returns:
            Quantized tensor.
        """
        # Clip data in range
        clipped_tensor = tf.keras.backend.clip(weight, self.min_range, self.max_range)
        # Quantize the data between min/max of quantization range.
        quantized_kernel = tf.keras.backend.round((clipped_tensor - self.min_range) / self.delta)
        return quantized_kernel * self.delta + self.min_range

    def __call__(self, inputs, training, weights, **kwargs):
        """
        Apply quantization to the input tensor.

        Args:
            inputs: Input tensor to be quantized.
            training: Whether the graph is currently training.
            weights: Dictionary of weights the quantizer can use to quantize the tensor. This contains the weights created in the `build` function.
            **kwargs: Additional variables which may be passed to the quantizer.

        Returns:
            Quantized tensor.
        """
        with tf.name_scope('WeightsUniformQuant'):
            return self.weight

    def get_config(self):
        """

        Returns: Configuration of this WeightsUniformQuantizer

        """
        cfg = {"nbits": self.nbits,
               "min_range": self.min_range.numpy(),
               "max_range": self.max_range.numpy(),
               "weight": self.weight.numpy(),
               "quantization_method": self.quantization_method
               }
        return cfg

    def build(self, tensor_shape: TensorShape, name: str, layer: Layer) -> dict:
        """
        Add variables under layer's scope.

        Args:
            tensor_shape: Shape of tensor which needs to be quantized.
            name: Name of tensor.
            layer: Layer to add variables to.

        Returns:
            Dictionary with new layer's variables.
        """
        return {}
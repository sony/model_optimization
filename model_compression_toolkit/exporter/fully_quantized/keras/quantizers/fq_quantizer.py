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
# ==============================================================================\

from typing import Dict, Any
import numpy as np
import tensorflow as tf
from keras.engine.base_layer import Layer
from tensorflow import TensorShape
from tensorflow_model_optimization.python.core.quantization.keras.quantizers import Quantizer

from model_compression_toolkit.core.common.target_platform import QuantizationMethod


class FakeQuantQuantizer(Quantizer):
    """
    Quantizer using TensorFlow fake quant layer to quantize activations.
    """

    def __init__(self,
                 nbits: int,
                 min_range: np.ndarray,
                 max_range: np.ndarray,
                 quantization_method: QuantizationMethod):
        """

        Args:
            nbits: Number of bits to quantize.
            min_range: Min quantization range.
            max_range: Max quantization range.
            quantization_method: Quantization method that is used (POT, Uniform, etc.)

        """
        self.nbits = nbits
        self.min_range = tf.Variable(min_range,
                                     trainable=False,
                                     dtype=tf.float32)
        self.max_range = tf.Variable(max_range,
                                     trainable=False,
                                     dtype=tf.float32)
        self.quantization_method = quantization_method


    def get_config(self) -> Dict[str, Any]:
        """

        Returns: Configuration of this FakeQuantQuantizer

        """
        return {"nbits": self.nbits,
                "min_range": self.min_range.numpy(),
                "max_range": self.max_range.numpy(),
                "quantization_method": self.quantization_method,
                }

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
        with tf.name_scope('FakeQuant'):
            return tf.quantization.fake_quant_with_min_max_vars(inputs,
                                                                min=self.min_range,
                                                                max=self.max_range,
                                                                num_bits=self.nbits)


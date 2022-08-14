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
from tensorflow_model_optimization.python.core.quantization.keras.quantizers import Quantizer

from model_compression_toolkit.core.common.quantization.quantizers.quantizers_helpers import fix_range_to_include_zero


class WeightsUniformQuantizer(Quantizer):
    def __init__(self,
                 nbits,
                 min_range,
                 max_range,
                 weight: tf.Tensor
                 ):

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

    def _uniform_quantize(self, weight):
        # Clip data in range
        clipped_tensor = tf.keras.backend.clip(weight, self.min_range, self.max_range)
        # Quantize the data between min/max of quantization range.
        quantized_kernel = tf.keras.backend.round((clipped_tensor - self.min_range) / self.delta)
        return quantized_kernel * self.delta + self.min_range

    def __call__(self, inputs, training, weights, **kwargs):
        with tf.name_scope('WeightsUniformQuant'):
            return self.weight

    def get_config(self):
        cfg = {"nbits": self.nbits,
               "min_range": self.min_range.numpy(),
               "max_range": self.max_range.numpy(),
               "weight": self.weight.numpy()
               }
        return cfg

    def build(self, tensor_shape, name, layer):
        return {}
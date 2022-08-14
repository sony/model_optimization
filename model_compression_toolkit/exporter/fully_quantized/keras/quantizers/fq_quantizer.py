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

import numpy as np
import tensorflow as tf
from tensorflow_model_optimization.python.core.quantization.keras.quantizers import Quantizer

from model_compression_toolkit.core.common.quantization.quantizers.quantizers_helpers import fix_range_to_include_zero


class FakeQuantQuantizer(Quantizer):

    def __init__(self,
                 nbits,
                 min_range,
                 max_range,
                 ):
        self.nbits = nbits
        self.min_range = tf.Variable(min_range,
                                     trainable=False,
                                     dtype=tf.float32)
        self.max_range = tf.Variable(max_range,
                                     trainable=False,
                                     dtype=tf.float32)


    def get_config(self):
        return {"nbits": self.nbits,
                "min_range": self.min_range.numpy(),
                "max_range": self.max_range.numpy()
                }

    def build(self, tensor_shape, name, layer):
        return {}

    def __call__(self, inputs, training, weights, **kwargs):
        with tf.name_scope('FakeQuant'):
            return tf.quantization.fake_quant_with_min_max_vars(inputs,
                                                                min=self.min_range,
                                                                max=self.max_range,
                                                                num_bits=self.nbits)


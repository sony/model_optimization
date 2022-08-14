import numpy as np
import tensorflow as tf
from tensorflow_model_optimization.python.core.quantization.keras.quantizers import Quantizer

from model_compression_toolkit.core.common.quantization.quantizers.quantizers_helpers import fix_range_to_include_zero


class UniformQuantizer(Quantizer):

    def __init__(self,
                 nbits,
                 min_range,
                 max_range,
                 ):
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

    def get_config(self):
        return {"nbits": self.nbits,
                "min_range": self.min_range.numpy(),
                "max_range": self.max_range.numpy()
                }

    def build(self, tensor_shape, name, layer):
        return {}

    def __call__(self, inputs, training, weights, **kwargs):
        with tf.name_scope('UniformQuant'):
            # Clip data in range
            clipped_tensor = tf.keras.backend.clip(inputs, self.min_range, self.max_range)
            # Quantize the data between min/max of quantization range.
            quantized_kernel = tf.keras.backend.round((clipped_tensor - self.min_range) / self.delta)
            quantized_kernel = quantized_kernel * self.delta + self.min_range
            return quantized_kernel


class WeightsUniformQuantizer(UniformQuantizer):
    def __init__(self,
                 nbits,
                 min_range,
                 max_range,
                 weight: tf.Tensor
                 ):
        super().__init__(nbits=nbits,
                         min_range=min_range,
                         max_range=max_range)

        # Quantize weights staticly and store it to return it during __call__
        self.weight = super(WeightsUniformQuantizer, self).__call__(weight,
                                                                    False,
                                                                    {})

    def __call__(self, inputs, training, weights, **kwargs):
        with tf.name_scope('WeightsUniformQuant'):
            return self.weight

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"weight": self.weight.numpy()})
        return cfg

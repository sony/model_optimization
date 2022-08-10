import tensorflow as tf
from tensorflow_model_optimization.python.core.quantization.keras.quantizers import Quantizer


class UniformQuantizer(Quantizer):

    def __init__(self,
                 nbits,
                 min_range,
                 max_range,
                 channel_axis: int = -1,
                 per_channel: bool = True):

        self.nbits = nbits

        self.min_range = tf.Variable(min_range,
                                     trainable=False,
                                     dtype=tf.float32)

        self.max_range = tf.Variable(max_range,
                                     trainable=False,
                                     dtype=tf.float32)

        self.channel_axis = channel_axis
        self.per_channel = per_channel
        self.delta = (self.max_range - self.min_range) / (2 ** self.nbits - 1)

    def get_config(self):
        return {"nbits": self.nbits,
                "min_range": self.min_range.numpy(),
                "max_range": self.max_range.numpy(),
                "channel_axis": self.channel_axis,
                "per_channel": self.per_channel}

    def build(self, tensor_shape, name, layer):
        return {}

    def __call__(self, inputs, training, weights, **kwargs):
        with tf.name_scope('UniformQuant'):
            # if self.per_channel:
            # Clip data in range
            clipped_tensor = tf.keras.backend.clip(inputs, self.min_range, self.max_range)
            # Quantize the data between min/max of quantization range.
            quantized_kernel = tf.keras.backend.round((clipped_tensor - self.min_range) / self.delta)
            quantized_kernel = quantized_kernel * self.delta + self.min_range
            # else:
            #     # Clip data in range
            #     clipped_tensor = tf.clip_by_value(inputs,
            #                                       clip_value_min=self.min_range,
            #                                       clip_value_max=self.max_range)
            #
            #     # Quantize the data between min/max of quantization range.
            #     quantized_kernel = tf.round((clipped_tensor - self.min_range) / delta)

            return quantized_kernel


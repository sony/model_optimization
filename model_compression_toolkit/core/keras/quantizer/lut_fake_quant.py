import numpy as np
import tensorflow as tf
from keras.layers import Layer

from model_compression_toolkit.core.common.constants import SIGNED, CLUSTER_CENTERS, EPS, \
    MULTIPLIER_N_BITS, THRESHOLD


class LUTFakeQuant(Layer):
    def __init__(self, quantization_params, **kwargs):
        super(LUTFakeQuant, self).__init__(**kwargs)
        self.quantization_params = quantization_params
        self.activation_is_signed = self.quantization_params.get(SIGNED)
        self.cluster_centers = self.quantization_params.get(CLUSTER_CENTERS)
        self.threshold = self.quantization_params.get(THRESHOLD) 

    def build(self, input_shape):
        super(LUTFakeQuant, self).build(input_shape)

    def call(self, input_data, **kwargs):
        if self.activation_is_signed is None or self.cluster_centers is None or self.threshold is None:
            return None

        _quant_output = self.lut_kmeans_quantizer(input_data)
        return _quant_output

    def lut_kmeans_quantizer(self, tensor_data) -> np.ndarray:
        tensor = self.int_quantization_with_threshold(tensor_data, self.threshold, MULTIPLIER_N_BITS)
        tensor = tf.expand_dims(tensor, -1)

        expanded_cluster_centers = self.cluster_centers.reshape([*[1 for _ in range(len(tensor.shape)-1)], -1])
        cluster_assignments = tf.argmin(tf.abs(tensor - expanded_cluster_centers), axis=-1)
        centers = tf.gather(self.cluster_centers.flatten(), cluster_assignments)

        quant_tensor = (centers / (2 ** (MULTIPLIER_N_BITS - int(self.activation_is_signed)))) * self.threshold

        return quant_tensor

    def int_quantization_with_threshold(self,
                                        data: np.ndarray,
                                        threshold: np.ndarray,
                                        n_bits: int,
                                        eps: float = EPS):

        if self.activation_is_signed:
            clip_max = 2 ** (n_bits - 1) - 1
            clip_min = -2 ** (n_bits - 1)
        else:
            clip_max = 2 ** n_bits - 1
            clip_min = 0

        return tf.clip_by_value((data / (threshold + eps)) * (2 ** (n_bits - int(self.activation_is_signed))),
                                clip_value_max=clip_max, clip_value_min=clip_min)

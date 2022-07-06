import numpy as np
import tensorflow as tf
from keras.layers import Layer

from model_compression_toolkit.core.common.constants import SIGNED, CLUSTER_CENTERS, SCALE_PER_CHANNEL, EPS, \
    MULTIPLIER_N_BITS


class LUTFakeQuant(Layer):
    def __init__(self, n_bits, quantization_params, **kwargs):
        super(LUTFakeQuant, self).__init__(**kwargs)
        self.quantization_params = quantization_params
        self.activation_is_signed = self.quantization_params.get(SIGNED)
        self.cluster_centers = self.quantization_params.get(CLUSTER_CENTERS)
        self.scales_per_channel = self.quantization_params.get(SCALE_PER_CHANNEL)

    def build(self, input_shape):
        super(LUTFakeQuant, self).build(input_shape)

    def call(self, input_data, **kwargs):
        if self.activation_is_signed is None or self.cluster_centers is None or self.scales_per_channel is None:
            return None

        _quant_output = self.lut_kmeans_quantizer(input_data,
                                                  self.activation_is_signed,
                                                  self.cluster_centers,
                                                  self.scales_per_channel)
        return _quant_output

    def lut_kmeans_quantizer(self,
                             tensor_data,
                             signed: bool,
                             cluster_centers: np.ndarray,
                             scales_per_channel: np.ndarray) -> np.ndarray:

        cluster_centers = np.round(cluster_centers)
        tensor = self.int_quantization_with_scale(tensor_data, scales_per_channel, MULTIPLIER_N_BITS, signed=signed)
        tensor = tf.expand_dims(tensor, -1)
        expanded_cluster_centers = cluster_centers.reshape([*[1 for _ in range(len(tensor.shape)-1)], -1])
        cluster_assignments = tf.argmin(tf.abs(tensor - expanded_cluster_centers), axis=-1)
        # shape_before_kmeans = tensor.shape
        # no_batch_tensor = tf.squeeze(tensor, [0])
        # cluster_assignments = self.kmeans_assign_clusters(cluster_centers, tf.reshape(no_batch_tensor, (-1, 1)))
        centers = tf.gather(cluster_centers.flatten(), cluster_assignments)

        quant_tensor = (centers / (2 ** (MULTIPLIER_N_BITS - 1))) * scales_per_channel

        return quant_tensor

    @staticmethod
    def int_quantization_with_scale(data: np.ndarray,
                                    scale: np.ndarray,
                                    n_bits: int,
                                    signed: bool = True,
                                    eps: float = EPS):
        return tf.clip_by_value(data / (scale + eps) * 2 ** (n_bits - 1),
                                clip_value_max=2 ** (n_bits - 1) - 1, clip_value_min=-2 ** (n_bits - 1))

    @staticmethod
    def kmeans_assign_clusters(cluster_centers: np.ndarray,
                               query: np.ndarray) -> np.ndarray:
        """
        Assign each data value in query with its closest cluster center point.
        Args:
            cluster_centers: the cluster centers to assign the query values.
            query: values for which to assign cluster centers.

        Returns: A tensor of indexes to the cluster centers that where assigned to each value in
                 the query tensor.

        """
        d0 = query.shape[0]
        d1 = cluster_centers.shape[0]
        query_ = tf.reshape(tf.repeat(query, d1), [d0, d1])
        cluster_centers_ = cluster_centers.repeat(d0).reshape(d1, d0).transpose(1, 0)
        return tf.argmin(tf.abs(query_ - cluster_centers_), axis=1)

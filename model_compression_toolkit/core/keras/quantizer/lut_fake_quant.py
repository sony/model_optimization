from typing import Tuple, Dict, Callable

import numpy as np
import tensorflow as tf
from keras.layers import Layer
from tensorflow.python.util.object_identity import Reference as TFReference

from model_compression_toolkit.constants import SIGNED, LUT_VALUES, EPS, \
    LUT_VALUES_BITWIDTH, THRESHOLD


def activation_lut_kmean_quantizer(activation_n_bits: int,
                                   quantization_params: Dict[str, np.ndarray]) -> Callable:
    """
    Builds a LUT quantizer for layer's activation using the provided params (threshold and clusters).
    It initiates a fake custom LUT layer that provides the quantizer function.

    Args:
        activation_n_bits: Number of bits to use for quantization (not used in this function).
        quantization_params: Dictionary of specific parameters for this quantization function.

    Returns:
        A fake LUT quantization node.
    """

    lut_fake_quant = LUTFakeQuant(quantization_params=quantization_params)
    return lambda x: lut_fake_quant(x)


class LUTFakeQuant(Layer):
    """
    A custom Keras layer for quantizing activation tensor with non-uniform quantization (using lookup table values).
    """

    def __init__(self, quantization_params: Dict[str, np.ndarray], **kwargs):
        super(LUTFakeQuant, self).__init__(**kwargs)
        self.quantization_params = quantization_params
        self.activation_is_signed = self.quantization_params.get(SIGNED)
        self.lut_values = self.quantization_params.get(LUT_VALUES)
        self.threshold = self.quantization_params.get(THRESHOLD)

    def build(self, input_shape: Tuple[int]):
        """
        Builds the layer.

        Args:
            input_shape: The layer's input shape.

        """
        super(LUTFakeQuant, self).build(input_shape)

    def call(self, input_data: TFReference, **kwargs) -> TFReference:
        """

        Args:
            input_data: A Keras input tensor.
            **kwargs: Optional arguments' dictionary.

        Returns: KerasTensor after applying a non-uniform fake quantization.

        """
        if self.activation_is_signed is None or self.lut_values is None or self.threshold is None:
            return None  # pragma: no cover

        _quant_output = self.lut_kmeans_quantizer(input_data)
        return _quant_output

    def lut_kmeans_quantizer(self, tensor_data: TFReference) -> TFReference:
        """
        Quantize a tensor using a non-uniform quantization based on the pre-defined kmeans clusters.
        1. Scales tensor_data with the threshold into 8-bit quantization range.
        2. Assigns cluster centers to each value.
        3. Scales back by multiplying the result by threshold and dividing with the quantization range max value.
        The result is the quantized tensor.

        Args:
            tensor_data: Input activation tensor.

        Returns: Quantized tensor.
        """

        tensor = self.int_quantization_with_threshold(tensor_data, LUT_VALUES_BITWIDTH)
        tensor = tf.expand_dims(tensor, -1)

        expanded_lut_values = self.lut_values.reshape([*[1 for _ in range(len(tensor.shape)-1)], -1])
        lut_values_assignments = tf.argmin(tf.abs(tensor - expanded_lut_values), axis=-1)
        centers = tf.gather(self.lut_values.flatten(), lut_values_assignments)

        quant_tensor = (centers / (2 ** (LUT_VALUES_BITWIDTH - int(self.activation_is_signed)))) * self.threshold

        return quant_tensor

    def int_quantization_with_threshold(self,
                                        data: TFReference,
                                        n_bits: int,
                                        eps: float = EPS) -> TFReference:
        """
        Divides data by threshold and quantize it to integers in the quantization range (depends on signed value).

        Args:
            data: tensor data.
            n_bits: number of bits that determines the quantization range.
            eps: Small value for numerical stability in division.

        Returns:
            Uniform Quantized tensor.

        """

        if self.activation_is_signed:
            clip_max = 2 ** (n_bits - 1) - 1
            clip_min = -2 ** (n_bits - 1)
        else:
            clip_max = 2 ** n_bits - 1
            clip_min = 0

        return tf.clip_by_value((data / (self.threshold + eps)) * (2 ** (n_bits - int(self.activation_is_signed))),
                                clip_value_max=clip_max, clip_value_min=clip_min)

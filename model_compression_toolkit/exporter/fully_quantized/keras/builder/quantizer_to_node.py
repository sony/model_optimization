from typing import List

from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.common.constants import THRESHOLD, SIGNED
from model_compression_toolkit.core.common.target_platform import QuantizationMethod
import numpy as np

from model_compression_toolkit.exporter.fully_quantized.keras.quantizers.fq_quantizer import FakeQuantQuantizer
from model_compression_toolkit.exporter.fully_quantized.keras.quantizers.uniform_quantizer import \
    WeightsUniformQuantizer


def calculate_delta(threshold: np.ndarray,
                     n_bits: int = 8,
                     signed: bool = False) -> np.ndarray:
    """
    Compute the step size of quantized values given the threshold, number of bits
    and whether its signed or unsigned.

    Args:
        threshold: Threshold to compute the step size according to.
        n_bits: Number of bits to compute the step size according to.
        signed: Whether quantization range is signed or not.

    Returns:
        Step size of quantized values according to a threshold, signedness and number of bits.
    """

    return threshold / (2 ** (n_bits - int(signed)))


def get_weights_quantizer_for_node(node: BaseNode, weights_attr :List[str]):

    assert node.final_weights_quantization_cfg is not None, f'Can not set quantizer for a node with no final ' \
                                                            f'weights quantization configuration'

    supported_quantizers = [QuantizationMethod.POWER_OF_TWO,
                            QuantizationMethod.SYMMETRIC,
                            QuantizationMethod.UNIFORM]

    node_w_qc = node.final_weights_quantization_cfg
    weights_quantization_method = node_w_qc.weights_quantization_method
    assert weights_quantization_method in supported_quantizers, \
        f'Fully quantized models are now supported for {supported_quantizers} quantization methods, but node ' \
        f'has {weights_quantization_method} quantization method'

    if weights_quantization_method in [QuantizationMethod.POWER_OF_TWO, QuantizationMethod.SYMMETRIC]:
        # TODO: Add assertion for POT case that thresholds are POT
        min_range = -node_w_qc.weights_quantization_params.get(THRESHOLD)
        max_range = node_w_qc.weights_quantization_params.get(THRESHOLD) - calculate_delta(
            node_w_qc.weights_quantization_params.get(THRESHOLD),
            n_bits=node_w_qc.weights_n_bits,
            signed=True)
    elif weights_quantization_method in [QuantizationMethod.UNIFORM]:
        min_range = node_w_qc.weights_quantization_params.get(RANGE_MIN)
        max_range = node_w_qc.weights_quantization_params.get(RANGE_MAX)
    else:
        raise NotImplemented

    assert len(weights_attr )==1, f'Currently, we support only one quantized weight per layer'
    return WeightsUniformQuantizer(node_w_qc.weights_n_bits,
                                   min_range,
                                   max_range,
                                   node.get_weights_by_keys(weights_attr[0]))


def get_activations_quantizer_for_node(node: BaseNode):

    assert node.final_activation_quantization_cfg is not None, f'Can not set quantizer for a node with no final ' \
                                                               f'activation quantization configuration'

    supported_quantizers = [QuantizationMethod.POWER_OF_TWO,
                            QuantizationMethod.SYMMETRIC,
                            QuantizationMethod.UNIFORM]

    node_act_qc = node.final_activation_quantization_cfg
    activation_quantization_method = node_act_qc.activation_quantization_method
    assert activation_quantization_method in supported_quantizers, \
        f'Fully quantized models are now supported for {supported_quantizers} quantization methods, but node ' \
        f'has {activation_quantization_method} quantization method'

    if activation_quantization_method in [QuantizationMethod.POWER_OF_TWO, QuantizationMethod.SYMMETRIC]:
        # TODO: Add assertion for POT case that thresholds are POT
        min_range = 0
        if node_act_qc.activation_quantization_params.get(SIGNED):
            min_range = -node_act_qc.activation_quantization_params.get(THRESHOLD)
        max_range = node_act_qc.activation_quantization_params.get(THRESHOLD) - calculate_delta(
            node_act_qc.activation_quantization_params.get(THRESHOLD),
            n_bits=node_act_qc.activation_n_bits,
            signed=node_act_qc.activation_quantization_params.get(SIGNED))
    else:
        raise NotImplemented

    return FakeQuantQuantizer(node_act_qc.activation_n_bits,
                              min_range,
                              max_range)

    # return UniformQuantizer(node_act_qc.activation_n_bits,
    #                         min_range,
    #                         max_range)
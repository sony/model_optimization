from tensorflow_model_optimization.python.core.quantization.keras.default_8bit.default_8bit_quantize_configs import \
    NoOpQuantizeConfig

from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.exporter.fully_quantized.keras.builder.quantizer_to_node import \
    get_weights_quantizer_for_node, get_activations_quantizer_for_node
from model_compression_toolkit.exporter.fully_quantized.keras.quantize_configs.activation_quantize_config import \
    ActivationQuantizeConfig
from model_compression_toolkit.exporter.fully_quantized.keras.quantize_configs.weights_activation_quantize_config \
    import \
    WeightsActivationQuantizeConfig
from model_compression_toolkit.exporter.fully_quantized.keras.quantize_configs.weights_quantize_config import \
    WeightsQuantizeConfig
from model_compression_toolkit.core.common.framework_info import FrameworkInfo


def get_quantization_config(node: BaseNode, fw_info:FrameworkInfo):
    if node.is_weights_quantization_enabled() and not node.is_activation_quantization_enabled():
        weight_attrs = fw_info.get_kernel_op_attributes(node.type)
        return WeightsQuantizeConfig(weight_attrs=weight_attrs,
                                     w_quantizer=get_weights_quantizer_for_node(node,
                                                                                weight_attrs))

    elif not node.is_weights_quantization_enabled() and node.is_activation_quantization_enabled():
        return ActivationQuantizeConfig(activation_quantizer=get_activations_quantizer_for_node(node))

    elif not node.is_weights_quantization_enabled() and not node.is_activation_quantization_enabled():
        return NoOpQuantizeConfig()

    weight_attrs = fw_info.get_kernel_op_attributes(node.type)
    return WeightsActivationQuantizeConfig(activation_quantizer=get_activations_quantizer_for_node(node),
                                           w_quantizer=get_weights_quantizer_for_node(node,
                                                                                      weight_attrs),
                                           weight_attrs=fw_info.get_kernel_op_attributes(node.type))
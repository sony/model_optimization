# from model_compression_toolkit import FrameworkInfo
# from model_compression_toolkit.core.common import BaseNode
# from model_compression_toolkit.core.common.quantization.node_quantization_config import NodeWeightsQuantizationConfig
# from model_compression_toolkit.core.common.target_platform import QuantizationMethod
# from model_compression_toolkit.core.quantizers.keras.uniform_quantizer import UniformQuantizer
# from model_compression_toolkit.core.quantizers.keras.weights_quantize_config import WeightsQuantizeConfig
#
#
# def get_weights_quantizer(node_w_qc:NodeWeightsQuantizationConfig):
#     if node_w_qc.weights_quantization_method == QuantizationMethod.POWER_OF_TWO:
#         w_quantizer = UniformQuantizer(node_w_qc.weights_n_bits,
#                          -node_w_qc.weights_quantization_params.get('threshold'),  # In weights assume it's signed
#                          node_w_qc.weights_quantization_params.get('threshold'),
#                                        node_w_qc.weights_channels_axis,
#                                        node_w_qc.weights_per_channel_threshold)
#     else:
#         raise NotImplemented
#     return w_quantizer
#
# def get_activation_quantizer(node_a_qc):
#
#     pass
#
# def generate_quantize_config(node:BaseNode,
#                              fw_info:FrameworkInfo):
#     assert node.final_weights_quantization_cfg is not None
#     w_q = get_weights_quantizer(node.final_weights_quantization_cfg)
#     return WeightsQuantizeConfig(w_q,
#                                  [attr for attr in fw_info.get_kernel_op_attributes(node.type) if attr is not None])

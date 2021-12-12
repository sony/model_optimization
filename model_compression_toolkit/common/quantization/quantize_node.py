# Copyright 2021 Sony Semiconductors Israel, Inc. All rights reserved.
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


import copy

from model_compression_toolkit import common
from model_compression_toolkit.common import Logger
from model_compression_toolkit.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.common.graph.base_node import BaseNode
from model_compression_toolkit.common.framework_info import FrameworkInfo
from model_compression_toolkit.common.quantization.node_quantization_config import NodeWeightsQuantizationConfig
from model_compression_toolkit.common.quantization.quantization_params_generation.qparams_weights_computation import \
    get_channels_axis


def get_quantized_kernel_by_weights_qc(fw_info:FrameworkInfo,
                                       n:BaseNode,
                                       weights_qc: NodeWeightsQuantizationConfig,
                                       fw_impl: FrameworkImplementation):
    """
    For a node and a weights quantization configuration, compute
    the quantized kernel of the node and return it and the input/output channels indices.

    Args:
        fw_info: A FrameworkInfo object Information needed for quantization about the specific framework (e.g., kernel channels indices, groups of layers by how they should be quantized, etc.).
        n: Node to quantize its kernel.
        weights_qc: Weight quantization configuration to use for the quantization.
        fw_impl: FrameworkImplementation with specific framework implementations.

    Returns:
        A quantized kernel of the node using a weights quantization configuration.
    """

    # If weights should be quantized per-channel but a kernel channels mapping is missing.
    if weights_qc.weights_per_channel_threshold and fw_info.kernel_channels_mapping is \
            None:
        common.Logger.warning(
            'Weights Per Channel Quantization requires channel mapping function but framework info '
            'does not contain one')
    output_channels_axis, input_channels_axis = get_channels_axis(weights_qc,
                                                                  fw_info,
                                                                  n.layer_class)

    Logger.debug(f'quantizing {n.name} with {weights_qc.weights_n_bits} bits')
    quantized_kernel = weights_qc.weights_quantization_fn(n.get_weights_by_keys(fw_impl.constants.KERNEL),
                                                          n_bits=weights_qc.weights_n_bits,
                                                          signed=True,
                                                          quantization_params=weights_qc.weights_quantization_params,
                                                          per_channel=weights_qc.weights_per_channel_threshold,
                                                          output_channels_axis=output_channels_axis)

    return quantized_kernel, (input_channels_axis, output_channels_axis)
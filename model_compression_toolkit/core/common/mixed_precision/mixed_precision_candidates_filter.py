# Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
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
import numpy as np

from model_compression_toolkit.core import ResourceUtilization, FrameworkInfo
from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.framework_quantization_capabilities import \
    FrameworkQuantizationCapabilities


def filter_candidates_for_mixed_precision(graph: Graph,
                                          target_resource_utilization: ResourceUtilization,
                                          fw_info: FrameworkInfo,
                                          fqc: FrameworkQuantizationCapabilities):
    """
    Filters out candidates in case of mixed precision search for only weights or activation compression.
    For instance, if running only weights compression - filters out candidates of activation configurable nodes
    such that only a single candidate would remain, with the bitwidth equal to the one defined in the matching layer's
    base config in the FQC.

    Note: This function modifies the graph inplace!

    Args:
        graph: A graph representation of the model to be quantized.
        target_resource_utilization: The resource utilization of the target device.
        fw_info: fw_info: Information needed for quantization about the specific framework.
        fqc: FrameworkQuantizationCapabilities object that describes the desired inference target platform.

    """

    tru = target_resource_utilization
    if tru.total_mem_restricted() or tru.bops_restricted():
        return

    if tru.weight_restricted() and not tru.activation_restricted():
        # Running mixed precision for weights compression only -
        # filter out candidates activation only configurable node
        activation_configurable_nodes = [n for n in graph.get_activation_configurable_nodes()]
        for n in activation_configurable_nodes:
            base_cfg_nbits = n.get_qco(fqc).base_config.activation_n_bits
            filtered_conf = [c for c in n.candidates_quantization_cfg if
                             c.activation_quantization_cfg.enable_activation_quantization and
                             c.activation_quantization_cfg.activation_n_bits == base_cfg_nbits]

            n.candidates_quantization_cfg = filtered_conf

    elif tru.activation_restricted() and not tru.weight_restricted():
        # Running mixed precision for activation compression only -
        # filter out candidates weights only configurable node
        weight_configurable_nodes = [n for n in graph.get_weights_configurable_nodes(fw_info)]
        for n in weight_configurable_nodes:
            kernel_attr = fw_info.get_kernel_op_attributes(n.type)[0]
            base_cfg_nbits = n.get_qco(fqc).base_config.attr_weights_configs_mapping[kernel_attr].weights_n_bits
            filtered_conf = [c for c in n.candidates_quantization_cfg if
                             c.weights_quantization_cfg.get_attr_config(kernel_attr).enable_weights_quantization and
                             c.weights_quantization_cfg.get_attr_config(kernel_attr).weights_n_bits == base_cfg_nbits]
            n.candidates_quantization_cfg = filtered_conf

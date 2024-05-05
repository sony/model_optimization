# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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
from typing import Callable, Any, Dict, Tuple

from model_compression_toolkit.constants import FLOAT_BITWIDTH, BITS_TO_BYTES
from model_compression_toolkit.core import FrameworkInfo, ResourceUtilization, CoreConfig
from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.graph.edge import EDGE_SINK_INDEX
from model_compression_toolkit.core.graph_prep_runner import graph_preparation_runner
from model_compression_toolkit.target_platform_capabilities.target_platform import TargetPlatformCapabilities, \
    QuantizationConfigOptions


def compute_resource_utilization_data(in_model: Any,
                                      representative_data_gen: Callable,
                                      core_config: CoreConfig,
                                      tpc: TargetPlatformCapabilities,
                                      fw_info: FrameworkInfo,
                                      fw_impl: FrameworkImplementation,
                                      transformed_graph: Graph = None,
                                      mixed_precision_enable: bool = True) -> ResourceUtilization:
    """
    Compute Resource Utilization information that can be relevant for defining target ResourceUtilization for mixed precision search.
    Calculates maximal activation tensor size, the sum of the model's weight parameters and the total memory combining both weights
    and maximal activation tensor size.

    Args:
        in_model:  Model to build graph from (the model that intended to be quantized).
        representative_data_gen: Dataset used for calibration.
        core_config: CoreConfig containing parameters of how the model should be quantized.
        tpc: TargetPlatformCapabilities object that models the inference target platform and
                                              the attached framework operator's information.
        fw_info: Information needed for quantization about the specific framework.
        fw_impl: FrameworkImplementation object with a specific framework methods implementation.
        transformed_graph: An internal graph representation of the input model. Defaults to None.
                            If no graph is provided, a graph will be constructed using the specified model.
        mixed_precision_enable: Indicates if mixed precision is enabled, defaults to True.
                                If disabled, computes resource utilization using base quantization
                                configurations across all layers.

    Returns:
        ResourceUtilization: An object encapsulating the calculated resource utilization computations.


    """

    # We assume that the resource_utilization_data API is used to compute the model resource utilization for
    # mixed precision scenario, so we run graph preparation under the assumption of enabled mixed precision.
    if transformed_graph is None:
        transformed_graph = graph_preparation_runner(in_model,
                                                     representative_data_gen,
                                                     core_config.quantization_config,
                                                     fw_info,
                                                     fw_impl,
                                                     tpc,
                                                     mixed_precision_enable=mixed_precision_enable)

    # Compute parameters sum
    weights_memory_bytes, weights_params = compute_nodes_weights_params(graph=transformed_graph, fw_info=fw_info)
    total_weights_params = 0 if len(weights_params) == 0 else sum(weights_params)

    # Compute max activation tensor
    activation_output_sizes_bytes, activation_output_sizes = compute_activation_output_sizes(graph=transformed_graph)
    max_activation_tensor_size = 0 if len(activation_output_sizes) == 0 else max(activation_output_sizes)

    # Compute total memory utilization - parameters sum + max activation tensor
    total_size = total_weights_params + max_activation_tensor_size

    # Compute BOPS utilization - total count of bit-operations for all configurable layers with kernel
    bops_count = compute_total_bops(graph=transformed_graph, fw_info=fw_info, fw_impl=fw_impl)
    bops_count = np.inf if len(bops_count) == 0 else sum(bops_count)

    return ResourceUtilization(weights_memory=total_weights_params,
                               activation_memory=max_activation_tensor_size,
                               total_memory=total_size,
                               bops=bops_count)


def compute_nodes_weights_params(graph: Graph, fw_info: FrameworkInfo) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the memory usage in bytes and the number of weight parameters for each node within a graph.
    Memory calculations are based on the maximum bit-width used for quantization per node.

    Args:
        graph: A finalized Graph object, representing the model structure.
        fw_info: FrameworkInfo object containing details about the specific framework's
                 quantization attributes for different layers' weights.

    Returns:
        A tuple containing two arrays:
            - The first array represents the memory in bytes for each node's weights when quantized at the maximal bit-width.
            - The second array represents the total number of weight parameters for each node.
    """
    weights_params = []
    weights_memory_bytes = []
    for n in graph.nodes:
        # TODO: when enabling multiple attribute quantization by default (currently,
        #  only kernel quantization is enabled) we should include other attributes memory in the sum of all
        #  weights memory.
        #  When implementing this, we should just go over all attributes in the node instead of counting only kernels.
        kernel_attr = fw_info.get_kernel_op_attributes(n.type)[0]
        if kernel_attr is not None and not n.reuse:
            kernel_candidates = n.get_all_weights_attr_candidates(kernel_attr)

            if len(kernel_candidates) > 0 and any([c.enable_weights_quantization for c in kernel_candidates]):
                max_weight_bits = max([kc.weights_n_bits for kc in kernel_candidates])
                node_num_weights_params = 0
                for attr in fw_info.get_kernel_op_attributes(n.type):
                    if attr is not None:
                        node_num_weights_params += n.get_weights_by_keys(attr).flatten().shape[0]

                weights_params.append(node_num_weights_params)

                # multiply num params by num bits and divide by BITS_TO_BYTES to convert from bits to bytes
                weights_memory_bytes.append(node_num_weights_params * max_weight_bits / BITS_TO_BYTES)

    return np.array(weights_memory_bytes), np.array(weights_params)

def compute_activation_output_sizes(graph: Graph) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes an array of the respective output tensor size and an array of the output tensor size in bytes for
    each node.

    Args:
        graph: A finalized Graph object, representing the model structure.

    Returns:
    A tuple containing two arrays:
        - The first array represents the size of each node's activation output tensor size in bytes,
          calculated using the maximal bit-width for quantization.
        - The second array represents the size of each node's activation output tensor size.


    """

    activation_outputs = []
    activation_outputs_bytes = []
    for n in graph.nodes:
        # Go over all nodes that have configurable activation.
        if n.has_activation_quantization_enabled_candidate():
            # Fetch maximum bits required for quantizing activations
            max_activation_bits = max([qc.activation_quantization_cfg.activation_n_bits for qc in n.candidates_quantization_cfg])
            node_output_size = n.get_total_output_params()
            activation_outputs.append(node_output_size)
            # Calculate activation size in bytes and append to list
            activation_outputs_bytes.append(node_output_size * max_activation_bits / BITS_TO_BYTES)

    return np.array(activation_outputs_bytes), np.array(activation_outputs)


def compute_total_bops(graph: Graph, fw_info: FrameworkInfo, fw_impl: FrameworkImplementation) -> np.ndarray:
    """
    Computes a vector with the respective Bit-operations count for each configurable node that includes MAC operations.
    The computation assumes that the graph is a representation of a float model, thus, BOPs computation uses 32-bit.

    Args:
        graph: Finalized Graph object.
        fw_info: FrameworkInfo object about the specific framework
            (e.g., attributes of different layers' weights to quantize).
        fw_impl: FrameworkImplementation object with a specific framework methods implementation.

    Returns: A vector of nodes' Bit-operations count.

    """

    bops = []

    # Go over all configurable nodes that have kernels.
    for n in graph.get_topo_sorted_nodes():
        if n.has_kernel_weight_to_quantize(fw_info):
            # If node doesn't have weights then its MAC count is 0, and we shouldn't consider it in the BOPS count.
            incoming_edges = graph.incoming_edges(n, sort_by_attr=EDGE_SINK_INDEX)
            assert len(incoming_edges) == 1, f"Can't compute BOPS metric for node {n.name} with multiple inputs."

            node_mac = fw_impl.get_node_mac_operations(n, fw_info)

            node_bops = (FLOAT_BITWIDTH ** 2) * node_mac
            bops.append(node_bops)

    return np.array(bops)


def requires_mixed_precision(in_model: Any,
                            target_resource_utilization: ResourceUtilization,
                            representative_data_gen: Callable,
                            core_config: CoreConfig,
                            tpc: TargetPlatformCapabilities,
                            fw_info: FrameworkInfo,
                            fw_impl: FrameworkImplementation) -> bool:
    """
    The function checks whether the model requires mixed precision to meet the requested target resource utilization.
    This is determined by whether the target memory usage of the weights is less than the available memory,
    the target maximum size of an activation tensor is less than the available memory,
    and the target number of BOPs is less than the available BOPs.
    If any of these conditions are met, the function returns True. Otherwise, it returns False.

    Args:
        in_model: The model to be evaluated.
        target_resource_utilization: The resource utilization of the target device.
        representative_data_gen: A function that generates representative data for the model.
        core_config: CoreConfig containing parameters of how the model should be quantized.
        tpc: TargetPlatformCapabilities object that models the inference target platform and
                                              the attached framework operator's information.
        fw_info: Information needed for quantization about the specific framework.
        fw_impl: FrameworkImplementation object with a specific framework methods implementation.

    Returns: A boolean indicating if mixed precision is needed.
    """
    is_mixed_precision = False
    transformed_graph = graph_preparation_runner(in_model,
                                                 representative_data_gen,
                                                 core_config.quantization_config,
                                                 fw_info,
                                                 fw_impl,
                                                 tpc,
                                                 mixed_precision_enable=False)
    # Compute max weights memory in bytes
    weights_memory_by_layer_bytes, _ = compute_nodes_weights_params(transformed_graph, fw_info)
    total_weights_memory_bytes = 0 if len(weights_memory_by_layer_bytes) == 0 else sum(weights_memory_by_layer_bytes)

    # Compute max activation tensor in bytes
    activation_output_sizes_bytes, _ = compute_activation_output_sizes(transformed_graph)
    max_activation_tensor_size_bytes = 0 if len(activation_output_sizes_bytes) == 0 else max(activation_output_sizes_bytes)

    # Compute BOPS utilization - total count of bit-operations for all configurable layers with kernel
    bops_count = compute_total_bops(graph=transformed_graph, fw_info=fw_info, fw_impl=fw_impl)
    bops_count = np.inf if len(bops_count) == 0 else sum(bops_count)

    is_mixed_precision |= target_resource_utilization.weights_memory < total_weights_memory_bytes
    is_mixed_precision |= target_resource_utilization.activation_memory < max_activation_tensor_size_bytes
    is_mixed_precision |= target_resource_utilization.total_memory < total_weights_memory_bytes + max_activation_tensor_size_bytes
    is_mixed_precision |= target_resource_utilization.bops < bops_count
    return is_mixed_precision

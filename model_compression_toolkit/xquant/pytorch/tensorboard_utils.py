#  Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==============================================================================
from mct_quantizers import PytorchActivationQuantizationHolder, PytorchQuantizationWrapper
from model_compression_toolkit.core.common import Graph, BaseNode
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from typing import Dict, Any, Callable, List

import torch

from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.pytorch.reader.reader import model_reader
from model_compression_toolkit.xquant.common.constants import XQUANT_REPR, INTERMEDIATE_SIMILARITY_METRICS_REPR, XQUANT_VAL, INTERMEDIATE_SIMILARITY_METRICS_VAL
from model_compression_toolkit.xquant.common.model_folding_utils import ModelFoldingUtils
from model_compression_toolkit.xquant.common.tensorboard_utils import TensorboardUtils

NODES_WITHOUT_CUT_INFO = [torch.fake_quantize_per_tensor_affine]

def is_wrapped_linear_op(quantized_model, node):
    # Check if a node in a torch fx graph represents a linear layer (conv2d/linear)
    # that is wrapped in the quantized model
    return hasattr(quantized_model, node.name.removesuffix('_layer')) and isinstance(
        getattr(quantized_model, node.name.removesuffix('_layer')), PytorchQuantizationWrapper)

class PytorchTensorboardUtils(TensorboardUtils):
    """
    Utility class for handling PyTorch models with TensorBoard. Inherits from TensorboardUtils.
    This class provides functionalities to display quantized model graphs on TensorBoard.
    """

    def __init__(self,
                 report_dir: str,
                 fw_impl: FrameworkImplementation):
        """
        Initialize the PytorchTensorboardUtils instance.

        Args:
            report_dir: Directory where the reports are stored.
            fw_impl: Implementation methods for the framework.
        """
        super().__init__(report_dir,
                         fw_impl)

    def get_graph_for_tensorboard_display(self,
                                          quantized_model: torch.nn.Module,
                                          similarity_metrics: Dict[str, Any],
                                          repr_dataset: Callable,
                                          quantized_model_metadata: Dict):
        """
        Get the graph to display on TensorBoard. The graph represents the quantized model
        with the similarity metrics that were measured.

        Args:
            quantized_model: The quantized model to be displayed on TensorBoard.
            similarity_metrics: Dictionary containing the collected similarity metrics values.
            repr_dataset: Callable that generates the representative dataset used during graph building.
            quantized_model_metadata (Dict): Metadata from the quantized model.

        Returns:
            The updated quantized model graph with similarity metrics embedded.
        """
        # Read the model and generate a graph representation
        quant_graph = model_reader(quantized_model,
                                   representative_data_gen=repr_dataset,
                                   to_tensor=self.fw_impl.to_tensor,
                                   to_numpy=self.fw_impl.to_numpy)

        if 'scheduling_info' in quantized_model_metadata:
            insert_cut_info_into_graph(quant_graph, quantized_model_metadata, quantized_model)

        # Iterate through each node in the graph
        for node in quant_graph.nodes:
            # Check and add similarity metrics for each node in the graph
            if node.name in similarity_metrics[INTERMEDIATE_SIMILARITY_METRICS_REPR].keys():
                node.framework_attr[XQUANT_REPR] = similarity_metrics[INTERMEDIATE_SIMILARITY_METRICS_REPR][f"{node.name}"]
            elif node.name.removesuffix("_layer") in similarity_metrics[INTERMEDIATE_SIMILARITY_METRICS_REPR].keys():
                node.framework_attr[XQUANT_REPR] = similarity_metrics[INTERMEDIATE_SIMILARITY_METRICS_REPR][
                    node.name.removesuffix("_layer")]

            # Check and add validation similarity metrics for each node in the graph
            if node.name in similarity_metrics[INTERMEDIATE_SIMILARITY_METRICS_VAL].keys():
                node.framework_attr[XQUANT_VAL] = similarity_metrics[INTERMEDIATE_SIMILARITY_METRICS_VAL][f"{node.name}"]
            elif node.name.removesuffix("_layer") in similarity_metrics[INTERMEDIATE_SIMILARITY_METRICS_VAL].keys():
                node.framework_attr[XQUANT_VAL] = similarity_metrics[INTERMEDIATE_SIMILARITY_METRICS_VAL][
                    node.name.removesuffix("_layer")]

        return quant_graph


def populate_fused_node_memory_elements(quantized_model_metadata: Dict[str, Any]) -> Dict[str, list]:
    """
    Populate a dictionary mapping fused node names to their corresponding memory elements.

    Args:
        quantized_model_metadata: Metadata containing scheduling information for the quantized model.

    Returns:
        dict: A dictionary with fused node names as keys and memory elements as values.
    """
    fused_node_to_memory_elements = {}

    for cut in quantized_model_metadata['scheduling_info']['cuts']:
        fused_node = cut['op_order'][-1]

        # Ignore dummy types
        if not fused_node.startswith('DummyType'):
            fused_node_to_memory_elements[fused_node] = cut['mem_elements']

    return fused_node_to_memory_elements


def assign_cut_info_to_node(node: BaseNode, memory_elements: List[dict]):
    """
    Assign cut memory elements and total size to a node's framework attributes.

    Args:
        node (Node): The node to which the memory elements and total size will be assigned.
        memory_elements (list): List of memory elements to be assigned to the node.
    """
    node.framework_attr['cut_memory_elements'] = [
        f"{mem_element['node_name']}_outTensor_{mem_element['node_output_index']}"
        for mem_element in memory_elements
    ]
    node.framework_attr['cut_total_size'] = sum(
        mem_element['total_size'] for mem_element in memory_elements
    )


def process_node_cut_info(node: BaseNode, fused_node_to_memory_elements: Dict[str, list], quantized_model_metadata: Dict[str, Any], quantized_model: torch.nn.Module):
    """
    Process and assign cut information for a given node based on metadata and fused nodes mapping.

    Args:
        node: The node to process.
        fused_node_to_memory_elements: Dictionary mapping fused nodes to memory elements.
        quantized_model_metadata: Metadata containing scheduling information for the quantized model.
        quantized_model: The quantized model.
    """
    node_name_without_suffix = node.name.removesuffix('_layer')
    fused_nodes_mapping = quantized_model_metadata['scheduling_info']['fused_nodes_mapping']

    if node.name in fused_node_to_memory_elements:
        # Directly assign cut info if node name is in fused_node_to_memory_elements
        assign_cut_info_to_node(node, fused_node_to_memory_elements[node.name])

    elif is_wrapped_linear_op(quantized_model, node) and node_name_without_suffix in fused_node_to_memory_elements:
        # Assign cut info if the node is a wrapped linear operation with a matching name without suffix
        assign_cut_info_to_node(node, fused_node_to_memory_elements[node_name_without_suffix])

    elif node.name in fused_nodes_mapping:
        # Assign cut info if the node name is in the fused nodes mapping
        original_node_name = fused_nodes_mapping[node.name]
        assign_cut_info_to_node(node, fused_node_to_memory_elements[original_node_name])

    elif is_wrapped_linear_op(quantized_model, node) and node_name_without_suffix in fused_nodes_mapping:
        # Assign cut info if the node is a wrapped linear operation and its name without suffix is in the fused nodes mapping
        original_node_name = fused_nodes_mapping[node_name_without_suffix]
        assign_cut_info_to_node(node, fused_node_to_memory_elements[original_node_name])


def insert_cut_info_into_graph(quant_graph: Graph,
                               quantized_model_metadata: Dict[str, Any],
                               quantized_model: torch.nn.Module):
    """
    Insert information about cut tensors into the graph nodes based on the provided metadata.

    Args:
        quant_graph: The graph representing the quantized model.
        quantized_model_metadata: Metadata containing scheduling information for the quantized model.
        quantized_model: The quantized model.
    """
    # Populate the mapping of fused nodes to memory elements
    fused_node_to_memory_elements = populate_fused_node_memory_elements(quantized_model_metadata)

    for node in quant_graph.nodes:
        # Skip nodes without cut information
        if node.type not in NODES_WITHOUT_CUT_INFO:
            process_node_cut_info(node, fused_node_to_memory_elements, quantized_model_metadata, quantized_model)



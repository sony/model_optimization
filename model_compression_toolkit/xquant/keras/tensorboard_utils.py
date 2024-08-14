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
from typing import Dict, Callable, Any

import keras

from mct_quantizers import KerasActivationQuantizationHolder, KerasQuantizationWrapper
from model_compression_toolkit.constants import MEM_ELEMENTS, CUTS, OP_ORDER, NODE_NAME, NODE_OUTPUT_INDEX, TOTAL_SIZE, FUSED_NODES_MAPPING
from model_compression_toolkit.core.common import Graph, BaseNode
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.framework_info import FrameworkInfo

from model_compression_toolkit.core.keras.reader.reader import model_reader

from model_compression_toolkit.xquant.common.constants import XQUANT_REPR, INTERMEDIATE_SIMILARITY_METRICS_REPR, \
    XQUANT_VAL, INTERMEDIATE_SIMILARITY_METRICS_VAL, CUT_MEMORY_ELEMENTS, CUT_TOTAL_SIZE
from model_compression_toolkit.xquant.common.tensorboard_utils import TensorboardUtils

NODES_WITHOUT_CUT_INFO = [KerasActivationQuantizationHolder]


class KerasTensorboardUtils(TensorboardUtils):
    """
    A utility class for handling TensorBoard operations specific to Keras models.
    This class extends the generic TensorboardUtils class and provides methods
    to facilitate the visualization of quantized models and their similarity metrics
    in TensorBoard.
    """

    def __init__(self, report_dir: str,
                 fw_info: FrameworkInfo,
                 fw_impl: FrameworkImplementation):
        """
        Initialize the KerasTensorboardUtils class with the given parameters.

        Args:
            report_dir (str): Directory where the TensorBoard files will be stored.
            fw_info (FrameworkInfo): Information about the framework being used.
            fw_impl (FrameworkImplementation): Implementation functions for the framework.
        """
        super().__init__(report_dir,
                         fw_info,
                         fw_impl)

    def get_graph_for_tensorboard_display(self,
                                          quantized_model: keras.Model,
                                          similarity_metrics: Dict[str, Dict[str, float]],
                                          repr_dataset: Callable,
                                          quantized_model_metadata: Dict) -> Graph:
        """
        Generate a graph suitable for TensorBoard display from the provided quantized model
        and similarity metrics.

        Args:
            quantized_model (keras.Model): The quantized Keras model for which the graph is to be created.
            similarity_metrics (Dict[str, Dict[str, float]]): A dictionary containing similarity metrics
                for different nodes in the model.
            repr_dataset (Callable): A function or callable that provides the representative dataset.
            quantized_model_metadata (Dict): Metadata from the quantized model.

        Returns:
            Graph: A graph object representing the quantized model, annotated with similarity metrics.
        """
        # Read the quantized model into a graph structure.
        quant_graph = model_reader(quantized_model)

        if 'scheduling_info' in quantized_model_metadata:
            insert_cut_info_into_graph(quant_graph, quantized_model_metadata)

        # Iterate over each node in the graph.
        for node in quant_graph.nodes:
            # Check if the node's name is in the similarity metrics for intermediate representation.
            if node.name in similarity_metrics[INTERMEDIATE_SIMILARITY_METRICS_REPR].keys():
                # If so, add the similarity metric for intermediate representation to the node's attributes.
                node.framework_attr[XQUANT_REPR] = similarity_metrics[INTERMEDIATE_SIMILARITY_METRICS_REPR][node.name]

            # Check if the node's name is in the similarity metrics for validation.
            if node.name in similarity_metrics[INTERMEDIATE_SIMILARITY_METRICS_VAL].keys():
                # If so, add the similarity metric for validation to the node's attributes.
                node.framework_attr[XQUANT_VAL] = similarity_metrics[INTERMEDIATE_SIMILARITY_METRICS_VAL][node.name]

        return quant_graph


def populate_fused_node_memory_elements(quantized_model_metadata: Dict[str, Any]) -> Dict[str, list]:
    """
    Populate a dictionary mapping fused node names to their corresponding memory elements.

    Args:
        quantized_model_metadata (dict): Metadata containing scheduling information for the quantized model.

    Returns:
        dict: A dictionary with fused node names as keys and memory elements as values.
    """
    fused_node_to_memory_elements = {}

    for cut in quantized_model_metadata['scheduling_info'][CUTS]:
        fused_node = cut[OP_ORDER][-1]

        # Ignore dummy types
        if not fused_node.startswith('DummyType'):
            fused_node_to_memory_elements[fused_node] = cut[MEM_ELEMENTS]

    return fused_node_to_memory_elements

def assign_cut_info_to_node(node: BaseNode, memory_elements: list):
    """
    Assign cut memory elements and total size to a node's attributes according to the
    tensors in the cut of this node.

    Args:
        node (Node): The node to which the memory elements and total size will be assigned.
        memory_elements (list): List of memory elements to be assigned to the node since they are in memory during this node inference.
    """
    node.framework_attr[CUT_MEMORY_ELEMENTS] = [
        f"{mem_element[NODE_NAME]}_outTensor_{mem_element[NODE_OUTPUT_INDEX]}"
        for mem_element in memory_elements
    ]
    node.framework_attr[CUT_TOTAL_SIZE] = sum(
        mem_element[TOTAL_SIZE] for mem_element in memory_elements
    )

def process_node_cut_info(node: BaseNode,
                          fused_node_to_memory_elements: Dict[str, list],
                          quantized_model_metadata: Dict[str, Any]):
    """
    Process and assign cut information for a given node based on metadata and fused nodes mapping.

    Args:
        node (Node): The node to process.
        fused_node_to_memory_elements (dict): Dictionary mapping fused nodes to memory elements.
        quantized_model_metadata (dict): Metadata containing scheduling information for the quantized model.
    """
    if node.name in fused_node_to_memory_elements:
        # Directly assign cut info if node name is in fused_node_to_memory_elements
        assign_cut_info_to_node(node, fused_node_to_memory_elements[node.name])

    elif node.name in quantized_model_metadata['scheduling_info'][FUSED_NODES_MAPPING]:
        # Assign cut info if the node name is in the fused nodes mapping
        original_node_name = quantized_model_metadata['scheduling_info'][FUSED_NODES_MAPPING][node.name]
        assign_cut_info_to_node(node, fused_node_to_memory_elements[original_node_name])

    elif node.type == KerasQuantizationWrapper:
        if node.framework_attr['layer']['config']['name'] in fused_node_to_memory_elements:
            # Assign cut info if the node is a KerasQuantizationWrapper with a matching layer name
            assign_cut_info_to_node(node, fused_node_to_memory_elements[node.framework_attr['layer']['config']['name']])

        elif node.framework_attr['layer']['config']['name'] in quantized_model_metadata['scheduling_info'][FUSED_NODES_MAPPING]:
            # Assign cut info if the node is a KerasQuantizationWrapper and its layer name is in the fused nodes mapping
            original_node_name = quantized_model_metadata['scheduling_info'][FUSED_NODES_MAPPING][node.framework_attr['layer']['config']['name']]
            assign_cut_info_to_node(node, fused_node_to_memory_elements[original_node_name])

def insert_cut_info_into_graph(quant_graph: Graph, quantized_model_metadata: Dict[str, Any]):
    """
    Insert information about cut tensors into the graph nodes based on the provided metadata.

    Args:
        quant_graph (Graph): The graph representing the quantized model.
        quantized_model_metadata (dict): Metadata containing scheduling information for the quantized model.
    """
    # Populate the mapping of fused nodes to memory elements
    fused_node_to_memory_elements = populate_fused_node_memory_elements(quantized_model_metadata)

    for node in quant_graph.nodes:
        # Skip nodes without cut information
        if node.type not in NODES_WITHOUT_CUT_INFO:
            process_node_cut_info(node,
                                  fused_node_to_memory_elements,
                                  quantized_model_metadata)


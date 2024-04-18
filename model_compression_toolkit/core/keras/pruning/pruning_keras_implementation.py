# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
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

from typing import List, Tuple, Dict

from model_compression_toolkit.core.common.pruning.pruning_framework_implementation import \
    PruningFrameworkImplementation
from model_compression_toolkit.core.common.pruning.pruning_section import PruningSection
from model_compression_toolkit.core.keras.keras_implementation import KerasImplementation
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.keras.constants import BIAS, GROUPS, FILTERS, UNITS, USE_BIAS
import keras

import numpy as np

from model_compression_toolkit.logger import Logger


class PruningKerasImplementation(KerasImplementation, PruningFrameworkImplementation):
    """
    Implementation of the PruningFramework for the Keras framework. This class provides
    concrete implementations of the abstract methods defined in PruningFrameworkImplementation
    for the Keras framework.
    """

    def prune_entry_node(self,
                         node: BaseNode,
                         output_mask: np.ndarray,
                         fw_info: FrameworkInfo):
        """
        Prunes the entry node of a model in Keras.

        Args:
            node (BaseNode): The entry node to be pruned.
            output_mask (np.ndarray): A numpy array representing the mask to be applied to the output channels.
            fw_info (FrameworkInfo): Framework-specific information object.

        """
        return _prune_keras_edge_node(node=node,
                                      mask=output_mask,
                                      fw_info=fw_info,
                                      is_exit_node=False)

    def prune_intermediate_node(self,
                                node: BaseNode,
                                input_mask: np.ndarray,
                                output_mask: np.ndarray,
                                fw_info: FrameworkInfo):
        """
        Prunes an intermediate node in a Keras model.

        Args:
            node (BaseNode): The intermediate node to be pruned.
            input_mask (np.ndarray): A numpy array representing the mask to be applied to the input channels.
            output_mask (np.ndarray): A numpy array representing the mask to be applied to the output channels.
            fw_info (FrameworkInfo): Framework-specific information object.

        """
        _edit_node_input_shape(input_mask, node)
        pruned_parameters = {}
        mask_bool = output_mask.astype(bool)
        for k, v in node.weights.items():
            # Apply the mask to the weights.
            pruned_parameters[k] = v.compress(mask_bool, axis=-1)
        node.weights = pruned_parameters

    def prune_exit_node(self,
                        node: BaseNode,
                        input_mask: np.ndarray,
                        fw_info: FrameworkInfo):
        """
        Prunes the exit node of a model in Keras.

        Args:
            node (BaseNode): The exit node to be pruned.
            input_mask (np.ndarray): A numpy array representing the mask to be applied to the input channels.
            fw_info (FrameworkInfo): Framework-specific information object.

        """
        return _prune_keras_edge_node(node=node,
                                      mask=input_mask,
                                      fw_info=fw_info,
                                      is_exit_node=True)

    def is_node_entry_node(self, node: BaseNode) -> bool:
        """
        Determines whether a node is an entry node in a Keras model.

        Args:
            node (BaseNode): The node to be checked.

        Returns:
            bool: Boolean indicating if the node is an entry node.
        """
        return _is_keras_node_pruning_section_edge(node)

    def is_node_exit_node(self,
                          node: BaseNode,
                          corresponding_entry_node: BaseNode,
                          fw_info: FrameworkInfo) -> bool:
        """
        Determines whether a node is an exit node in a Keras model.

        Args:
            node (BaseNode): The node to be checked.
            corresponding_entry_node (BaseNode): The entry node of the pruning section that is checked.
            fw_info (FrameworkInfo): Framework-specific information object.

        Returns:
            bool: Boolean indicating if the node is an exit node.
        """
        return _is_keras_node_pruning_section_edge(node) and PruningSection.has_matching_channel_count(node,
                                                                                                       corresponding_entry_node,
                                                                                                       fw_info)

    def is_node_intermediate_pruning_section(self, node: BaseNode) -> bool:
        """
        Determines whether a node is part of the intermediate section in the pruning process of a Keras model.

        Args:
            node (BaseNode): The node to be checked.

        Returns:
            bool: Boolean indicating if the node is part of the intermediate pruning section.
        """
        # Nodes that are not Conv2D, Conv2DTranspose, DepthwiseConv2D, or Dense are considered intermediate.
        return node.type not in [keras.layers.DepthwiseConv2D,
                                 keras.layers.Conv2D,
                                 keras.layers.Conv2DTranspose,
                                 keras.layers.Dense]

    def attrs_oi_channels_info_for_pruning(self,
                                           node: BaseNode,
                                           fw_info: FrameworkInfo) -> Dict[str, Tuple[int, int]]:
        """
        Retrieves the attributes of a given node along with the output/input (OI) channel axis
        for each attribute used to prune these attributes.

        Not all attributes of a node are directly associated with both input and output channels.
        For example, bias vectors in convolutional layers are solely related to the number of output
        channels and do not have a corresponding input channel dimension.
        In cases like that, None is returned in the tuple of axis for such attributes.

        For kernel operations (like convolutions), the function identifies the output and input
        channel axis based on framework-specific information.
        For non-kernel operations, it defaults to setting the last axis as the output
        channel axis, assuming no specific input channel axis.

        Args:
            node (BaseNode): The node from the computational graph.
            fw_info (FrameworkInfo): Contains framework-specific information and utilities.

        Returns:
            Dict[str, Tuple[int, int]]: A dictionary where each key is an attribute name (like 'kernel' or 'bias')
            and each value is a tuple representing the output and input channel axis indices respectively.
        """

        attributes_with_axis = {}
        if fw_info.is_kernel_op(node.type):
            kernel_attributes = fw_info.get_kernel_op_attributes(node.type)
            if kernel_attributes is None or len(kernel_attributes)==0:
                Logger.critical(f"Expected kernel attributes for operation for node type {node.type}, found None or empty.")

            for attr in kernel_attributes:
                attributes_with_axis[attr] = fw_info.kernel_channels_mapping.get(node.type)

            # Bias is a vector at the length of the number of output channels.
            # For this reason, input channel axis is irrelevant to the bias attribute.
            attributes_with_axis[BIAS] = (0, None)
        else:
            # We have several assumptions here:
            # 1. For intermediate nodes, we prune all nodes' weights.
            # 2. The output channel axis is the last axis of this attribute.
            # 3. The input channel axis is irrelevant since these attributes are pruned only by
            #    their output channels.
            for attr in list(node.weights.keys()):
                attributes_with_axis[attr] = (-1, None)

        return attributes_with_axis


def _is_keras_node_pruning_section_edge(node: BaseNode) -> bool:
    """
    Determines if a Keras node is an edge of a pruning section.

    In the context of pruning, an 'edge' node is a layer that can potentially be pruned.
    This function identifies such nodes based on their type and attributes. Specifically,
    Conv2D and Conv2DTranspose layers with 'groups' attribute set to 1, and Dense layers
    are considered as edges for pruning sections.

    Args:
        node (BaseNode): The node to be evaluated.

    Returns:
        bool: True if the node is an edge of a pruning section, False otherwise.
    """

    # Check if the node is a Conv2D or Conv2DTranspose layer with groups set to 1.
    if node.is_match_type(keras.layers.Conv2D) or node.is_match_type(keras.layers.Conv2DTranspose):
        return node.framework_attr[GROUPS] == 1
    return node.is_match_type(keras.layers.Dense)


def _prune_keras_edge_node(node: BaseNode,
                           mask: np.ndarray,
                           fw_info: FrameworkInfo,
                           is_exit_node: bool):
    """
    Prunes the given Keras node by applying the mask to the node's weights (kernels and biases).
    This function can handle both entry and exit nodes by specifying the is_exit_node parameter.

    Args:
        node: The node to be pruned.
        mask: The pruning mask to be applied.
        fw_info: Framework-specific information object.
        is_exit_node: A boolean indicating whether the node is an exit node.

    """

    # Retrieve the kernel attribute and the axes to prune.
    kernel_attr = fw_info.get_kernel_op_attributes(node.type)[0]
    io_axis = fw_info.kernel_channels_mapping.get(node.type)
    axis_to_prune = io_axis[int(is_exit_node)]
    kernel = node.get_weights_by_keys(kernel_attr)
    # Convert mask to boolean.
    mask_bool = mask.astype(bool)

    pruned_kernel = kernel.compress(mask_bool, axis=axis_to_prune)
    node.set_weights_by_keys(name=kernel_attr, tensor=pruned_kernel)

    if not is_exit_node and node.framework_attr[USE_BIAS]:
        # Prune the bias if applicable and it's an entry node.
        bias = node.get_weights_by_keys(BIAS)
        pruned_bias = bias.compress(mask_bool)
        node.set_weights_by_keys(name=BIAS, tensor=pruned_bias)

    if not is_exit_node:
        # Update 'filters' or 'units' attributes for entry node Conv2D/Conv2DTranspose layers.
        if node.is_match_type(keras.layers.Conv2D) or node.is_match_type(keras.layers.Conv2DTranspose):
            node.framework_attr[FILTERS] = int(np.sum(mask))
        elif node.is_match_type(keras.layers.Dense):
            node.framework_attr[UNITS] = int(np.sum(mask))

    if is_exit_node:
        # Adjust the input shape for the last node in the section.
        _edit_node_input_shape(mask_bool, node)


def _edit_node_input_shape(input_mask: np.ndarray,
                           node: BaseNode):
    """
    Adjusts the input shape of a node based on the given input mask.

    This function modifies the input shape of the given node to reflect the pruning
    that has taken place. It updates the last dimension of the node's input shape
    to match the number of channels that remain after pruning.

    Args:
        input_mask (np.ndarray): A binary array where 1 indicates the channel is kept and 0 means pruned.
        node (BaseNode): The node whose input shape needs to be adjusted.
    """
    # Start with the current input shape of the node.
    new_input_shape = list(node.input_shape)

    # Adjust the last dimension of the shape to match the number of unpruned (retained) channels.
    # This is done by summing the mask, as each '1' in the mask represents a retained channel.
    new_input_shape[-1] = int(np.sum(input_mask))

    # Update the node's input shape with the new dimensions.
    node.input_shape = tuple(new_input_shape)


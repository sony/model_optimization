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

from typing import Tuple, Dict

from model_compression_toolkit.core.common.pruning.pruning_framework_implementation import \
    PruningFrameworkImplementation
from model_compression_toolkit.core.common.pruning.pruning_section import PruningSection
from model_compression_toolkit.core.pytorch.pytorch_implementation import PytorchImplementation
from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.pytorch.constants import BIAS, GROUPS, OUT_CHANNELS, OUT_FEATURES, NUM_FEATURES, \
    IN_CHANNELS, IN_FEATURES, NUM_PARAMETERS
import torch

import numpy as np

from model_compression_toolkit.logger import Logger


class PruningPytorchImplementation(PytorchImplementation, PruningFrameworkImplementation):
    """
    Implementation of the PruningFramework for the Pytorch framework. This class provides
    concrete implementations of the abstract methods defined in PruningFrameworkImplementation
    for the Pytorch framework.
    """

    def prune_entry_node(self,
                         node: BaseNode,
                         output_mask: np.ndarray):
        """
        Prunes the entry node of a model in Pytorch.

        Args:
            node (BaseNode): The entry node to be pruned.
            output_mask (np.ndarray): A numpy array representing the mask to be applied to the output channels.

        """
        return _prune_pytorch_edge_node(node=node,
                                        mask=output_mask,
                                        is_exit_node=False)

    def prune_intermediate_node(self,
                                node: BaseNode,
                                input_mask: np.ndarray,
                                output_mask: np.ndarray):
        """
        Prunes an intermediate node in a Pytorch model.

        Args:
            node (BaseNode): The intermediate node to be pruned.
            input_mask (np.ndarray): A numpy array representing the mask to be applied to the input channels.
            output_mask (np.ndarray): A numpy array representing the mask to be applied to the output channels.

        """
        # TODO (reuvenp/liord): Address handling of node parameters that can be either a single value across all channels or distinct per channel, e.g., PReLU. Consider developing a structured approach.
        pruning_en = True
        _edit_node_input_shape(node, input_mask)
        pruned_parameters = {}
        mask_bool = output_mask.astype(bool)
        node.weights = pruned_parameters
        if node.is_match_type(torch.nn.BatchNorm2d):
            node.framework_attr[NUM_FEATURES] = int(np.sum(input_mask))
        elif node.is_match_type(torch.nn.PReLU):
            if node.framework_attr[NUM_PARAMETERS] > 1:
                node.framework_attr[NUM_PARAMETERS] = int(np.sum(input_mask))
            else:
                pruning_en = False

        if pruning_en:
            for k, v in node.weights.items():
                # Apply the mask to the weights.
                pruned_parameters[k] = v.compress(mask_bool, axis=-1)

    def prune_exit_node(self,
                        node: BaseNode,
                        input_mask: np.ndarray):
        """
        Prunes the exit node of a model in Pytorch.

        Args:
            node (BaseNode): The exit node to be pruned.
            input_mask (np.ndarray): A numpy array representing the mask to be applied to the input channels.

        """
        return _prune_pytorch_edge_node(node=node,
                                        mask=input_mask,
                                        is_exit_node=True)

    def is_node_entry_node(self, node: BaseNode) -> bool:
        """
        Determines whether a node is an entry node in a Pytorch model.

        Args:
            node (BaseNode): The node to be checked.

        Returns:
            bool: Boolean indicating if the node is an entry node.
        """
        return _is_pytorch_node_pruning_section_edge(node)

    def is_node_exit_node(self,
                          node: BaseNode,
                          corresponding_entry_node: BaseNode) -> bool:
        """
        Determines whether a node is an exit node in a Pytorch model.

        Args:
            node (BaseNode): The node to be checked.
            corresponding_entry_node (BaseNode): The entry node of the pruning section that is checked.

        Returns:
            bool: Boolean indicating if the node is an exit node.
        """
        return _is_pytorch_node_pruning_section_edge(node) and PruningSection.has_matching_channel_count(node,
                                                                                                         corresponding_entry_node)

    def is_node_intermediate_pruning_section(self, node: BaseNode) -> bool:
        """
        Determines whether a node is part of the intermediate section in the pruning process of a Pytorch model.

        Args:
            node (BaseNode): The node to be checked.

        Returns:
            bool: Boolean indicating if the node is part of the intermediate pruning section.
        """
        # Nodes that are not Conv2d, ConvTranspose2d, or Linear are considered intermediate.
        # For PReLU prune attributes only if there is a parameter per channel
        return node.type not in [torch.nn.Conv2d,
                                 torch.nn.ConvTranspose2d,
                                 torch.nn.Linear]

    def attrs_oi_channels_info_for_pruning(self,
                                           node: BaseNode) -> Dict[str, Tuple[int, int]]:
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

        Returns:
            Dict[str, Tuple[int, int]]: A dictionary where each key is an attribute name (like 'weight' or 'bias')
            and each value is a tuple representing the output and input channel axis indices respectively.
        """

        attributes_with_axis = {}
        if node.is_kernel_op:
            attributes_with_axis[node.kernel_attr] = (node.channel_axis.output, node.channel_axis.input)

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
                # If the number of float parameters is 1 or less - is the case where
                # we have one parameter for all channels. For this case, we don't
                # want to prune the parameter.
                if node.get_num_parameters()[1] <= 1:
                    attributes_with_axis[attr] = (None, None)
                else:
                    attributes_with_axis[attr] = (-1, None)

        return attributes_with_axis


def _is_pytorch_node_pruning_section_edge(node: BaseNode) -> bool:
    """
    Determines if a Pytorch node is an edge of a pruning section.

    In the context of pruning, an 'edge' node is a layer that can potentially be pruned.
    This function identifies such nodes based on their type and attributes. Specifically,
    Conv2d and ConvTranspose2d layers with 'groups' attribute set to 1, and Linear layers
    are considered as edges for pruning sections.

    Args:
        node (BaseNode): The node to be evaluated.

    Returns:
        bool: True if the node is an edge of a pruning section, False otherwise.
    """

    # Check if the node is a Conv2D or Conv2DTranspose layer with groups set to 1.
    if node.is_match_type(torch.nn.Conv2d) or node.is_match_type(torch.nn.ConvTranspose2d):
        return node.framework_attr[GROUPS] == 1
    return node.is_match_type(torch.nn.Linear)


def _prune_pytorch_edge_node(node: BaseNode,
                             mask: np.ndarray,
                             is_exit_node: bool):
    """
    Prunes the given Pytorch node by applying the mask to the node's weights (weights and biases).
    This function can handle both entry and exit nodes by specifying the is_exit_node parameter.

    Args:
        node (BaseNode): The node to be pruned.
        mask (np.ndarray): The pruning mask to be applied.
        is_exit_node (bool): A boolean indicating whether the node is an exit node.

    """

    # Retrieve the kernel attribute and the axes to prune.
    axis_to_prune = node.channel_axis.input if is_exit_node else node.channel_axis.output
    kernel = node.get_weights_by_keys(node.kernel_attr)
    # Convert mask to boolean.
    mask_bool = mask.astype(bool)

    pruned_kernel = kernel.compress(mask_bool, axis=axis_to_prune)
    node.set_weights_by_keys(name=node.kernel_attr, tensor=pruned_kernel)

    if not is_exit_node and node.framework_attr[BIAS]:
        # Prune the bias if applicable and it's an entry node.
        bias = node.get_weights_by_keys(BIAS)
        pruned_bias = bias.compress(mask_bool)
        node.set_weights_by_keys(name=BIAS, tensor=pruned_bias)

    if not is_exit_node:
        # Update 'out_channels' or 'out_features' attributes for entry nodes
        # Conv2d,ConvTranspose2d / Linear layers.
        if node.is_match_type(torch.nn.Conv2d) or node.is_match_type(torch.nn.ConvTranspose2d):
            node.framework_attr[OUT_CHANNELS] = int(np.sum(mask))
        elif node.is_match_type(torch.nn.Linear):
            node.framework_attr[OUT_FEATURES] = int(np.sum(mask))
        else:
            Logger.critical(f"{node.type} is currently not supported"
                             f"as an edge node in a pruning section")

    if is_exit_node:
        if node.is_match_type(torch.nn.Conv2d) or node.is_match_type(torch.nn.ConvTranspose2d):
            node.framework_attr[IN_CHANNELS] = int(np.sum(mask))
        elif node.is_match_type(torch.nn.Linear):
            node.framework_attr[IN_FEATURES] = int(np.sum(mask))
        else:
            Logger.critical(f"{node.type} is currently not supported"
                             f"as an edge node in a pruning section")
        # Adjust the input shape for the last node in the section.
        _edit_node_input_shape(node, mask_bool)


def _edit_node_input_shape(node: BaseNode,
                           input_mask: np.ndarray):
    """
    Adjusts the input shape of a node based on the given input mask.

    This function modifies the input shape of the given node to reflect the pruning
    that has taken place. It updates the last dimension of the node's input shape
    to match the number of channels that remain after pruning.

    Args:
        node (BaseNode): The node whose input shape needs to be adjusted.
        input_mask (np.ndarray): A binary array where 1 indicates the channel is kept and 0 means pruned.
    """
    # Start with the current input shape of the node.
    new_input_shape = list(node.input_shape)

    # Adjust the last dimension of the shape to match the number of unpruned (retained) channels.
    # This is done by summing the mask, as each '1' in the mask represents a retained channel.
    channel_axis = node.out_channel_axis
    new_input_shape[0][channel_axis] = int(np.sum(input_mask))

    # Update the node's input shape with the new dimensions.
    node.input_shape = tuple(new_input_shape)

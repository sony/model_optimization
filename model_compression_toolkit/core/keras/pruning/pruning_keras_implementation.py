from typing import List, Tuple

from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.common.pruning.pruning_framework_implementation import \
    PruningFrameworkImplementation
from model_compression_toolkit.core.keras.keras_implementation import KerasImplementation
from model_compression_toolkit.core.keras.pruning.attributes_info import get_keras_node_attributes_with_io_axis
from model_compression_toolkit.core.keras.pruning.check_node_role import is_keras_node_intermediate_pruning_section, \
    is_keras_entry_node, is_keras_exit_node
from model_compression_toolkit.core.keras.pruning.prune_keras_node import (prune_keras_exit_node,
                                                                           prune_keras_entry_node, \
                                                                           prune_keras_intermediate_node)
import numpy as np

class PruningKerasImplementation(KerasImplementation, PruningFrameworkImplementation):
    """
    Implementation of the PruningFramework for the Keras framework. This class provides
    concrete implementations of the abstract methods defined in PruningFrameworkImplementation
    for the Keras framework.
    """

    def prune_entry_node(self, node: BaseNode, output_mask: np.ndarray, fw_info: FrameworkInfo):
        """
        Prunes the entry node of a model in Keras.

        Args:
            node: The entry node to be pruned.
            output_mask: A numpy array representing the mask to be applied to the output channels.
            fw_info: Framework-specific information object.

        Returns:
            The result from the pruning operation.
        """
        return prune_keras_entry_node(node, output_mask, fw_info)

    def prune_intermediate_node(self, node: BaseNode, input_mask: np.ndarray, output_mask: np.ndarray, fw_info: FrameworkInfo):
        """
        Prunes an intermediate node in a Keras model.

        Args:
            node: The intermediate node to be pruned.
            input_mask: A numpy array representing the mask to be applied to the input channels.
            output_mask: A numpy array representing the mask to be applied to the output channels.
            fw_info: Framework-specific information object.

        Returns:
            The result from the pruning operation.
        """
        return prune_keras_intermediate_node(node, input_mask, output_mask, fw_info)

    def prune_exit_node(self, node: BaseNode, input_mask: np.ndarray, fw_info: FrameworkInfo):
        """
        Prunes the exit node of a model in Keras.

        Args:
            node: The exit node to be pruned.
            input_mask: A numpy array representing the mask to be applied to the input channels.
            fw_info: Framework-specific information object.

        Returns:
            The result from the pruning operation.
        """
        return prune_keras_exit_node(node, input_mask, fw_info)

    def is_node_entry_node(self, node: BaseNode):
        """
        Determines whether a node is an entry node in a Keras model.

        Args:
            node: The node to be checked.

        Returns:
            Boolean indicating if the node is an entry node.
        """
        return is_keras_entry_node(node)

    def is_node_exit_node(self, node: BaseNode, corresponding_entry_node: BaseNode, fw_info: FrameworkInfo):
        """
        Determines whether a node is an exit node in a Keras model.

        Args:
            node: The node to be checked.
            corresponding_entry_node: A related entry node to assist in the determination.

        Returns:
            Boolean indicating if the node is an exit node.
        """
        return is_keras_exit_node(node, corresponding_entry_node, fw_info)

    def is_node_intermediate_pruning_section(self, node):
        """
        Determines whether a node is part of the intermediate section in the pruning process of a Keras model.

        Args:
            node: The node to be checked.

        Returns:
            Boolean indicating if the node is part of the intermediate pruning section.
        """
        return is_keras_node_intermediate_pruning_section(node)

    def get_node_attributes_with_io_axis(self,
                                             node: BaseNode,
                                             fw_info: FrameworkInfo) -> List[Tuple[str, int]]:
        """
        Gets the attributes of a node and the axis for each attribute's output channels dimension.

        Args:
            node (BaseNode): The node for which attributes and their output channel axis are required.
            fw_info (FrameworkInfo): Framework-specific information containing details about layers and attributes.

        Returns:
            List[Tuple[str, int]]: A list of tuples where each tuple contains an attribute name and the axis
                                   of the output channels for that attribute.
        """
        return get_keras_node_attributes_with_io_axis(node, fw_info)

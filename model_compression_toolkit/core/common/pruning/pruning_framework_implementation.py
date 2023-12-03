from typing import List, Tuple

from abc import abstractmethod

from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
import numpy as np


class PruningFrameworkImplementation(FrameworkImplementation):

    @abstractmethod
    def prune_entry_node(self,
                         node: BaseNode,
                         output_mask: np.ndarray,
                         fw_info: FrameworkInfo):
        """
        Abstract method to prune an entry node in the model.

        Args:
            node: The node to be pruned.
            output_mask: A numpy array representing the mask to be applied to the output channels.
            fw_info: Framework-specific information.

        Raises:
            NotImplemented: If the method is not implemented in the subclass.
        """
        raise NotImplemented(f'{self.__class__.__name__} have to implement the '
                             f'framework\'s prune_entry_node method.')  # pragma: no cover

    @abstractmethod
    def prune_intermediate_node(self,
                                node: BaseNode,
                                input_mask: np.ndarray,
                                output_mask: np.ndarray,
                                fw_info: FrameworkInfo):
        """
        Abstract method to prune an intermediate node in the model.

        Args:
            node: The node to be pruned.
            input_mask: Mask to be applied to the input channels.
            output_mask: Mask to be applied to the output channels.
            fw_info: Framework-specific information.

        Raises:
            NotImplemented: If the method is not implemented in the subclass.
        """
        raise NotImplemented(f'{self.__class__.__name__} have to implement the '
                             f'framework\'s prune_intermediate_node method.')  # pragma: no cover

    @abstractmethod
    def prune_exit_node(self,
                        node: BaseNode,
                        input_mask: np.ndarray,
                        fw_info: FrameworkInfo):
        """
        Abstract method to prune an exit node in the model.

        Args:
            node: The node to be pruned.
            input_mask: Mask to be applied to the input channels.
            fw_info: Framework-specific information.

        Raises:
            NotImplemented: If the method is not implemented in the subclass.
        """
        raise NotImplemented(f'{self.__class__.__name__} have to implement the '
                             f'framework\'s prune_exit_node method.')  # pragma: no cover

    @abstractmethod
    def is_node_entry_node(self,
                           node: BaseNode):
        """
        Abstract method to determine if a given node is an entry node.

        Args:
            node: The node to be checked.

        Returns:
            bool: True if the node is an entry node, False otherwise.

        Raises:
            NotImplemented: If the method is not implemented in the subclass.
        """
        raise NotImplemented(f'{self.__class__.__name__} have to implement the '
                             f'framework\'s is_node_entry_node method.')  # pragma: no cover

    @abstractmethod
    def is_node_exit_node(self,
                          node: BaseNode,
                          dual_entry_node: BaseNode):
        """
        Abstract method to determine if a given node is an exit node.

        Args:
            node: The node to be checked.
            dual_entry_node: Another node to be used in the determination process.

        Returns:
            bool: True if the node is an exit node, False otherwise.

        Raises:
            NotImplemented: If the method is not implemented in the subclass.
        """
        raise NotImplemented(f'{self.__class__.__name__} have to implement the '
                             f'framework\'s is_node_exit_node method.')  # pragma: no cover

    @abstractmethod
    def is_node_intermediate_pruning_section(self,
                                             node):
        """
        Abstract method to determine if a given node is in the intermediate section of pruning.

        Args:
            node: The node to be checked.

        Returns:
            bool: True if the node is in the intermediate pruning section, False otherwise.

        Raises:
            NotImplemented: If the method is not implemented in the subclass.
        """
        raise NotImplemented(f'{self.__class__.__name__} have to implement the '
                             f'framework\'s is_node_intermediate_pruning_section method.')  # pragma: no cover

    def get_node_attributes_with_io_axis(self, node: BaseNode, fw_info: FrameworkInfo):
        """
        Gets the attributes of a node and the axis for each attribute's output channels dimension.

        Args:
            node (BaseNode): The node for which attributes and their output channel axis are required.
            fw_info (FrameworkInfo): Framework-specific information containing details about layers and attributes.

        Returns:
            List[Tuple[str, int]]: A list of tuples where each tuple contains an attribute name and the axis
                                   of the output channels for that attribute.
        """
        raise NotImplemented(f'{self.__class__.__name__} have to implement the '
                             f'framework\'s get_node_attributes_with_output_axis method.')  # pragma: no cover
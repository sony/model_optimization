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
                          corresponding_entry_node: BaseNode,
                          fw_info: FrameworkInfo):

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

    def get_node_attributes_with_oi_axis(self, node: BaseNode, fw_info: FrameworkInfo) -> Dict[str, Tuple[int, int]]:
        """
        Gets the attributes of a node and the axis for each attribute's output/input
        channels dimension.

        Args:
            node (BaseNode): The node for which attributes and their oi channel axis are required.
            fw_info (FrameworkInfo): Framework-specific information containing details about
            layers and attributes.

        Returns:
            Dict[str, Tuple[int, int]]: A dict of the node's attributes their oi axis.

        """
        raise NotImplemented(f'{self.__class__.__name__} have to implement the '
                             f'framework\'s get_node_attributes_with_output_axis method.')  # pragma: no cover
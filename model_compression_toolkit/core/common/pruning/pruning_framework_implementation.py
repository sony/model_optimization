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
                         output_mask: np.ndarray):
        """
        Abstract method to prune an entry node in the model.

        Args:
            node: The node to be pruned.
            output_mask: A numpy array representing the mask to be applied to the output channels.

        Raises:
            NotImplemented: If the method is not implemented in the subclass.
        """
        raise NotImplemented(f'{self.__class__.__name__} have to implement the '
                             f'framework\'s prune_entry_node method.')  # pragma: no cover

    @abstractmethod
    def prune_intermediate_node(self,
                                node: BaseNode,
                                input_mask: np.ndarray,
                                output_mask: np.ndarray):
        """
        Abstract method to prune an intermediate node in the model.

        Args:
            node: The node to be pruned.
            input_mask: Mask to be applied to the input channels.
            output_mask: Mask to be applied to the output channels.

        Raises:
            NotImplemented: If the method is not implemented in the subclass.
        """
        raise NotImplemented(f'{self.__class__.__name__} have to implement the '
                             f'framework\'s prune_intermediate_node method.')  # pragma: no cover

    @abstractmethod
    def prune_exit_node(self,
                        node: BaseNode,
                        input_mask: np.ndarray):
        """
        Abstract method to prune an exit node in the model.

        Args:
            node: The node to be pruned.
            input_mask: Mask to be applied to the input channels.

        Raises:
            NotImplemented: If the method is not implemented in the subclass.
        """
        raise NotImplemented(f'{self.__class__.__name__} have to implement the '
                             f'framework\'s prune_exit_node method.')  # pragma: no cover

    @abstractmethod
    def is_node_entry_node(self,
                           node: BaseNode) -> bool:
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
                          corresponding_entry_node: BaseNode) -> bool:

        raise NotImplemented(f'{self.__class__.__name__} have to implement the '
                             f'framework\'s is_node_exit_node method.')  # pragma: no cover

    @abstractmethod
    def is_node_intermediate_pruning_section(self,
                                             node: BaseNode) -> bool:
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

    def attrs_oi_channels_info_for_pruning(self, node: BaseNode) -> Dict[str, Tuple[int, int]]:
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
            Dict[str, Tuple[int, int]]: A dictionary where each key is an attribute name (like 'kernel' or 'bias')
            and each value is a tuple representing the output and input channel axis indices respectively.
        """
        raise NotImplemented(f'{self.__class__.__name__} have to implement the '
                             f'framework\'s attrs_oi_channels_info_for_pruning method.')  # pragma: no cover
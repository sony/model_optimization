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
from abc import abstractmethod, ABC
from model_compression_toolkit.core.common import BaseNode
import numpy as np


class BaseImportanceMetric(ABC):
    """
    Interface for implementing importance metrics used for pruning SIMD groups.
    """
    @abstractmethod
    def get_entry_node_to_simd_score(self, entry_nodes: List[BaseNode]) -> Tuple[
        Dict[BaseNode, np.ndarray], Dict[BaseNode, List[np.ndarray]]]:
        """
        Compute SIMD scores for each group of channels for a list of entry nodes.
        Group the channels into SIMD groups, and compute a score for each SIMD group.

        Args:
            entry_nodes (List[BaseNode]): Entry nodes of pruning sections in the graph.

        Returns:
            Tuple[Dict, Dict]: Tuple of two dictionaries. The first is a dictionary of entry nodes to
            numpy arrays where each element is an importance score for the SIMD group. The second
            dictionary maps each node to a list of numpy arrays where each numpy array is the indices
            of channels in a group.
        """
        raise NotImplemented(f'{self.__class__.__name__} have to implement the '
                             f'framework\'s get_entry_node_to_simd_score method.')  # pragma: no cover

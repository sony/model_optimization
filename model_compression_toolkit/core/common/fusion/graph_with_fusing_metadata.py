#  Copyright 2025 Sony Semiconductor Israel, Inc. All rights reserved.
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

import types

from functools import wraps

from typing import Any, Iterator

from model_compression_toolkit.core.common import BaseNode, Graph
from model_compression_toolkit.core.common.fusion.fusing_info import FusingInfo


class FusedLayerType:
    """
    Used to represent the type of fused layers, since __name__
    is accessed when the graph is displayed.
    """
    def __init__(self):
        self.__name__ = 'FusedLayer'


class GraphWithFusingMetadata:
    def __init__(self, graph: Graph, fusing_info: FusingInfo):
        """
        Initialize with a graph and its fusing information.

        Args:
            graph: The neural network graph (e.g., a networkx.DiGraph or similar).
            fusing_info: Dict mapping fused operation IDs to sets of node objects.
        """
        assert isinstance(graph, Graph)
        self._internal_graph = graph
        self._fusing_info = fusing_info
        self._fusing_info.validate(graph)  # Ensure initial consistency
        # TODO: temp disable activation quantization to keep similar functionality. This will be removed in the future
        self._disable_nodes_activation_quantization()

    # We added __getstate__ and __setstate__ to FusedGraph to fix a recursion error during copy.deepcopy. Without
    # these, deepcopy endlessly traverses attributes via __getattr__, causing a loop. Now, __getstate__ defines what
    # to copy (self._graph and self._fusing_info), and __setstate__ rebuilds the object, ensuring a clean copy
    # without recursion, assuming Graph and FusingInfo are copyable.
    def __getstate__(self):
        """
        Define how the object is serialized for copying.
        Returns a dictionary of the essential attributes.
        """
        self._fusing_info.validate(self._internal_graph)
        return self.__dict__.copy()

    def __setstate__(self, state):
        """
        Reconstruct the object from the serialized state.

        Args:
            state: Dictionary containing the serialized attributes.
        """
        self.__dict__.update(state)
        self._fusing_info.validate(self._internal_graph)

    def __getattr__(self, name: str) -> Any:
        """
        Delegate attribute access to the underlying graph if not found in FusedGraph.

        Ensures that if the accessed attribute is a callable (e.g., a method like remove_node),
        it is wrapped so that the fusing information is validated after execution.
        Non-callable attributes are returned directly without validation.

        Args:
            name: The name of the attribute being accessed.

        Returns:
            The attribute or a wrapped method from self._graph.

        Raises:
            AttributeError: If the attribute doesn't exist in self._graph.
        """

        # TODO: Optimize validation by restricting it to known modifying methods to improve efficiency. For now,
        #  validating after every method call ensures correctness. In the
        #  future, define explicit modification methods (e.g., remove_node)
        #  in FusedGraph for better efficiency.

        attr = getattr(self._internal_graph, name)
        # Only wrap methods or functions, excluding properties and descriptors
        if isinstance(attr, (types.MethodType, types.FunctionType)):
            @wraps(attr)
            def wrapper(*args, **kwargs):
                result = attr(*args, **kwargs)
                self._fusing_info.validate(self._internal_graph)
                return result

            return wrapper

        return attr

    def __iter__(self) -> Iterator[BaseNode]:
        """
        Make FusedGraph iterable by delegating to the underlying graph's iterator.

        This allows FusedGraph to be used in contexts expecting an iterable of nodes,
        such as topological_sort, without requiring changes to external code.

        Returns:
            An iterator over the nodes in the underlying graph.
        """
        return iter(self._internal_graph)

    def __getitem__(self, key: Any) -> Any:
        """
        Delegate subscripting to the underlying graph.

        This enables FusedGraph to support dictionary-like access (e.g., graph[node][child])
        as required by operations like topological_generations in NetworkX, maintaining
        compatibility with code expecting a subscriptable Graph object.

        Args:
            key: The key (e.g., node) to look up in the graph.

        Returns:
            The value associated with the key in the underlying graph.

        Raises:
            KeyError: If the key doesn't exist in self._graph.
        """
        return self._internal_graph[key]

    def update_fusing_info(self, new_fusing_info: FusingInfo):
        self._fusing_info = new_fusing_info

    def get_internal_graph(self):
        """Return the original graph."""
        return self._internal_graph

    def get_fusing_info(self):
        """Return the fusing information."""
        return self._fusing_info

    def is_part_of_fused_op(self, node):
        """Check if a node is part of any fused operation."""
        return self._fusing_info.is_node_in_fused_op(node)

    def _disable_nodes_activation_quantization(self):
        """
        Disable activation for non-quantization needed due to fusion
        Args:
            nodes: nodes to update their activation quantization
        """
        # TODO: temp disable activation quantization to keep similar functionality. This will be removed in the future
        nodes_to_disable = self._fusing_info.get_nodes_to_disable_act_quantization()
        for node in nodes_to_disable:
            for qc in node.candidates_quantization_cfg:
                qc.activation_quantization_cfg.enable_activation_quantization = False

    def validate(self):
        return self._fusing_info.validate(self._internal_graph)
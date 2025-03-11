# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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
from dataclasses import dataclass, field

from typing import List, Set

from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.common.graph.memory_graph.memory_element import MemoryElements


@dataclass(frozen=True)
class Cut:
    """
    A Cut object that contains a set of ordered nodes and their memory elements.

    Args:
        op_order: A list of the cut's nodes (model layers), ordered by their addition to the cut (first-to-last).
        op_record: A (unordered) set of the nodes in the cut.
        mem_elements: MemoryElements object which represents the activation tensors of the cut's nodes.
    """
    op_order: List[BaseNode]
    op_record: Set[BaseNode]
    mem_elements: MemoryElements

    _sorted_elements_signature: str = field(init=False, default=None)

    @property
    def sorted_elements_signature(self):
        if self._sorted_elements_signature is None:
            object.__setattr__(self, '_sorted_elements_signature',
                               '_'.join(sorted([e.node_name for e in self.mem_elements.elements])))
        return self._sorted_elements_signature

    def memory_size(self) -> float:
        """
        Returns: The total memory size of the cut.
        """

        return self.mem_elements.total_size

    def get_record_names(self) -> Set[str]:
        """
        Builds a set of the cut nodes' names.

        Returns: a set with the nodes' names.
        """

        return {op.name for op in self.op_record}

    def __eq__(self, other) -> bool:
        """
        Overrides the class equality method.
        Two Cuts are equal if they contain the same memory elements.

        Args:
            other: An object to compare the current object to.

        Returns: True if the two objects are equal. False otherwise.

        """
        if isinstance(other, Cut):
            return self.mem_elements == other.mem_elements
        return False  # pragma: no cover

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return f"<Cut: Nodes={[e.node_name for e in self.mem_elements.elements]}, size={self.memory_size()}>"  # pragma: no cover

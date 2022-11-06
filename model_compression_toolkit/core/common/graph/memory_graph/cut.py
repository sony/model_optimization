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
from typing import List, Set

from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.common.graph.memory_graph.memory_element import MemoryElements


class Cut:
    """
    A Cut object that contains a set of ordered nodes and their memory elements.
    """

    def __init__(self, op_order: List[BaseNode], op_record: Set[BaseNode], mem_elements: MemoryElements):
        """
        Args:
            op_order: A list of the cut's nodes (model layers), ordered by their addition to the cut (first-to-last).
            op_record: A (unordered) set of the nodes in the cut.
            mem_elements: MemoryElements object which represents the activation tensors of the cut's nodes.
        """

        self.op_order = op_order
        self.op_record = op_record
        self.mem_elements = mem_elements

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
        return False

    def __hash__(self):
        return hash((frozenset(self.op_order), frozenset(self.op_record), self.mem_elements))
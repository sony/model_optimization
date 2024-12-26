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
from typing import Set, Tuple, Any
import numpy as np


class ActivationMemoryTensor:
    """
    A representation of an activation output tensor of a model's layer.
    """

    def __init__(self, shape: Tuple[Any], node_name: str, node_output_index: int, init_size_to_zero: bool = False):
        """
        Args:
            shape: The shape of the activation tensor.
            node_name: The name of the node which its output is represented by the object.
            node_output_index: The index of this tensor in the node's outputs list.
            init_size_to_zero: Whether to initialize the memory tensor size to 0 or not.
        """

        # remove batch size (first element) from output shape. If the shape is a list then remove the first
        # axis. If shape a vector (e.g. output of size) then set the shape minus 1 to ignore the batch value.
        if len(shape) == 1:
            self.shape = [] if shape[0] is None else [shape[0] - 1]
        else:
            self.shape = shape[1:]
        # The total size of a tensor is considered to be the number of elements in the tensor
        self.total_size = self._get_tensor_total_size() if not init_size_to_zero else 0

        self.node_name = node_name
        self.node_output_index = node_output_index

    def _get_tensor_total_size(self) -> np.ndarray:
        """
        Returns: The total number of parameters in an activation tensor.
        """

        assert all([x is not None for x in self.shape])
        return np.prod(self.shape)


class MemoryElements:
    """
    An object which represents a set of activation tensors and their memory size.
    """

    def __init__(self, elements: Set[ActivationMemoryTensor], total_size: float):
        """
        Args:
            elements: A set of  ActivationMemoryTensor (the memory elements)
            total_size: The total number of parameters of the given tensors.
        """
        self.elements = elements
        self.total_size = total_size

    def add_element(self, new_element: ActivationMemoryTensor):
        """
        Adding an element to the set.

        Args:
            new_element: The element to add.

        """
        self.elements.add(new_element)
        self.total_size += new_element.total_size

    def add_elements_set(self, new_elements_set: Set[ActivationMemoryTensor]):
        """
        Adding a set of elements to the set.

        Args:
            new_elements_set: The elements to add.

        """
        self.elements.update(new_elements_set)
        self.total_size += sum([e.total_size for e in new_elements_set])

    def __eq__(self, other) -> bool:
        """
        Overrides the class equality method.
        Two MemoryElements objects are equal if they contain the same elements.

        Args:
            other: An object to compare the current object to.

        Returns: True if the two objects are equal. False otherwise.

        """
        if isinstance(other, MemoryElements):
            # MemoryElements are equal if they contain the exact same elements sets
            return self.elements == other.elements
        return False

    def __hash__(self):
        return hash((frozenset(self.elements)))

    def __copy__(self):
        """
        Overrides the class copy method.
        Creates a new set with the same elements that are in the copied object.

        Returns: A new MemoryElements object with a copied set of elements.

        """
        return MemoryElements({elm for elm in self.elements}, self.total_size)

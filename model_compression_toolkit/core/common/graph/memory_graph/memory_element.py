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
    def __init__(self, shape: Tuple[Any], node_name: str, node_output_index: int, total_size: float = -1):

        # remove batch size (first element) from output shape
        self.shape = shape[1:]
        self.total_size = self._get_tensor_total_size() if total_size == -1 else total_size

        self.node_name = node_name
        self.node_output_index = node_output_index

    def _get_tensor_total_size(self):
        assert all([x is not None for x in self.shape])
        return np.prod(self.shape)


class MemoryElements:
    def __init__(self, elements: Set[ActivationMemoryTensor], total_size: float):
        self.elements = elements
        self.total_size = total_size

    def add_element(self, new_element: ActivationMemoryTensor):
        self.elements.add(new_element)
        self.total_size += new_element.total_size

    def add_elements_set(self, new_elements_set: Set[ActivationMemoryTensor]):
        self.elements.update(new_elements_set)
        self.total_size += sum([e.total_size for e in new_elements_set])

    def __copy__(self):
        return MemoryElements({elm for elm in self.elements}, self.total_size)

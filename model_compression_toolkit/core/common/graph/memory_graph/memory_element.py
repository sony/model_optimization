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


class MemoryElement:
    def __init__(self, size):
        self.size = size


class MemoryElements:
    def __init__(self, elements, total_size):
        self.elements = elements
        self.total_size = total_size

    def add_element(self, new_element):
        self.elements.add(new_element)
        self.total_size += new_element.size

    def add_elements_set(self, new_elements_set):
        self.elements.update(new_elements_set)
        self.total_size += sum([e.size for e in new_elements_set])

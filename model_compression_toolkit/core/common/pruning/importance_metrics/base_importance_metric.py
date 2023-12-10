from typing import List

from abc import abstractmethod, ABC

from model_compression_toolkit.core.common import BaseNode

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

class BaseImportanceMetric(ABC):
    @abstractmethod
    def get_entry_node_to_simd_score(self, entry_nodes: List[BaseNode]):
        raise NotImplemented(f'{self.__class__.__name__} have to implement the '
                             f'framework\'s get_entry_node_to_simd_score method.')  # pragma: no cover





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

from abc import ABC, abstractmethod
from typing import List, Any, Dict

from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.hessian.hessian_config import HessianConfig


class HessianCalculator(ABC):

    def __init__(self,
                 graph: Graph,
                 config: HessianConfig,
                 input_images: List[Any],
                 fw_impl):
        self.graph = graph
        self.config = config
        self.input_images = input_images
        self.fw_impl = fw_impl

    @abstractmethod
    def compute(self):
        raise NotImplemented

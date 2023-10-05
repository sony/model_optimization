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
from typing import List

from enum import Enum

from model_compression_toolkit.core.common import BaseNode


class TraceHessianMode(Enum):
    """
    Enumeration representing the mode in which the Hessian is
    computed (w.r.t weights or activations of interest points).
    """
    WEIGHTS = 0
    ACTIVATIONS = 1


class TraceHessianGranularity(Enum):
    """
    Granularity of the Hessian computation.
    """
    PER_ELEMENT = 0
    PER_OUTPUT_CHANNEL = 1
    PER_TENSOR = 2


class TraceHessianRequest:
    """
    Configuration class for Hessian request.
    """

    def __init__(self,
                 mode: TraceHessianMode,
                 granularity: TraceHessianGranularity,
                 target_node: BaseNode,
                 ):
        """

        Args:
            mode: Determines whether to compute Hessian based on activations or weights
            granularity: Specifies the granularity (element, layer, channel) of Hessian computation
            target_node: Node to compute its trace hessian.

        """
        self.mode = mode  # activations or weights
        self.granularity = granularity  # per element, per layer, per channel
        self.target_node = target_node # TODO: extend it list of nodes

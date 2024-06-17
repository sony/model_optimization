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


class HessianMode(Enum):
    """
    Enum representing the mode for Hessian information computation.

    This determines whether the Hessian's approximation is computed w.r.t weights or w.r.t activations.
    Note: This is not the actual Hessian but an approximation.
    """
    WEIGHTS = 0         # Hessian approximation based on weights
    ACTIVATION = 1     # Hessian approximation based on activations


class HessianInfoGranularity(Enum):
    """
    Enum representing the granularity level for Hessian information computation.

    This determines the number the Hessian approximations is computed for some node.
    Note: This is not the actual Hessian but an approximation.
    """
    PER_ELEMENT = 0
    PER_OUTPUT_CHANNEL = 1
    PER_TENSOR = 2


class TraceHessianRequest:
    """
    Request configuration for the trace of the Hessian approximation.

    This class defines the parameters for the approximation of the trace of the Hessian matrix.
    It specifies the mode (weights/activations), granularity (element/channel/tensor), and the target node.
    Note: This does not compute the actual Hessian's trace but approximates it.
    """

    def __init__(self,
                 mode: HessianMode,
                 granularity: HessianInfoGranularity,
                 target_nodes: List,
                 ):
        """
        Attributes:
            mode (HessianMode): Mode of Hessian's trace approximation (w.r.t weights or activations).
            granularity (HessianInfoGranularity): Granularity level for the approximation.
            target_nodes (List[BaseNode]): The node in the float graph for which the Hessian's trace approximation is targeted.
        """

        self.mode = mode  # w.r.t activations or weights
        self.granularity = granularity  # per element, per layer, per channel
        self.target_nodes = target_nodes

    def __eq__(self, other):
        # Checks if the other object is an instance of TraceHessianRequest
        # and then checks if all attributes are equal.
        return isinstance(other, TraceHessianRequest) and \
               self.mode == other.mode and \
               self.granularity == other.granularity and \
               self.target_nodes == other.target_nodes

    def __hash__(self):
        # Computes the hash based on the attributes.
        # The use of a tuple here ensures that the hash is influenced by all the attributes.
        return hash((self.mode, self.granularity, tuple(self.target_nodes)))
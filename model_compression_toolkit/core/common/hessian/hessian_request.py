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


class HessianMode(Enum):
    """
    Enumeration representing the mode in which the Hessian is
    computed (w.r.t weights or activations of interest points).
    """
    WEIGHTS = 0
    ACTIVATIONS = 1


class HessianGranularity(Enum):
    """
    Granularity of the Hessian computation.
    """
    PER_ELEMENT = 0
    PER_OUTPUT_CHANNEL = 1
    PER_TENSOR = 2


class HessianRequest:
    """
    Configuration class for Hessian request.
    """

    def __init__(self,
                 mode: HessianMode,
                 granularity: HessianGranularity,
                 target_node: BaseNode,
                 ):
        """

        Args:
            mode: Determines whether to compute Hessian based on activations or weights
            granularity: Specifies the granularity (element, layer, channel) of Hessian computation
            target_node: Node for computation

        """
        self.mode = mode  # activations or weights
        self.granularity = granularity  # per element, per layer, per channel
        self.target_node = target_node # TODO: extend it list of nodes


# class MultiNodeHessianRequest:
#
#     def __init__(self,
#                  mode: HessianMode,
#                  granularity: HessianGranularity,
#                  target_nodes: List[BaseNode],
#                  ):
#
#         self.mode = mode
#         self.granularity = granularity
#         self.target_nodes = target_nodes
#
#         multi_request_by_node = []
#         for target_node in target_nodes:
#             request = HessianRequest(mode=mode,
#                                      granularity=granularity,
#                                      target_node=target_node)
#             multi_request_by_node.append(request)
#         self.multi_request_by_node = multi_request_by_node


    # def __eq__(self, other):
    #     """
    #     Checks equality of two HessianConfig objects.
    #     """
    #     if isinstance(other, HessianConfig):
    #         return (self.mode == other.mode and
    #                 self.granularity == other.granularity and
    #                 self.nodes_names_for_hessian_computation == other.nodes_names_for_hessian_computation and
    #                 self.alpha == other.alpha and
    #                 self.num_iterations == other.num_iterations and
    #                 self.norm_weights == other.norm_weights and
    #                 self.search_output_replacement == other.search_output_replacement)
    #     return False
    #
    # def __hash__(self):
    #     """
    #     Computes the hash of the HessianConfig object for dictionary usage or other hashing requirements.
    #     """
    #     return hash((self.mode,
    #                  self.granularity,
    #                  tuple(self.nodes_names_for_hessian_computation),
    #                  self.alpha,
    #                  self.num_iterations,
    #                  self.norm_weights,
    #                  self.search_output_replacement))

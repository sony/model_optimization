# Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
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

from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.matchers.node_matcher import BaseNodeMatcher


class ManualBitWidthSelection:
    """
   Class to encapsulate the manual bit width selection configuration for a specific filter.

   Attributes:
       filter (BaseNodeMatcher): The filter used to select nodes for bit width manipulation.
       bit_width (int): The bit width to be applied to the selected nodes.
   """
    def __init__(self,
                 filter: BaseNodeMatcher,
                 bit_width: int):
        self.filter = filter
        self.bit_width = bit_width


class BitWidthConfig:
    """
    Class to manage manual bit-width configurations.

    Attributes:
        manual_bit_width_selection_list (List[ManualBitWidthSelection]): A list of ManualBitWidthSelection objects defining manual bit-width configurations.
    """
    def __init__(self,
                 manual_bit_width_selection_list: List[ManualBitWidthSelection] = None):
        self.manual_bit_width_selection_list = [] if manual_bit_width_selection_list is None else manual_bit_width_selection_list

    def __repr__(self):
        # Used for debugging, thus no cover.
        return str(self.__dict__)  # pragma: no cover

    def set_manual_activation_bit_width(self, filter: BaseNodeMatcher, bit_width: int):
        """
        Add a manual bit-width selection to the configuration.

        Args:
            filter (BaseNodeMatcher): The filter used to select nodes for bit-width manipulation.
            bit_width (int): The bit width to be applied to the selected nodes.
        """
        self.manual_bit_width_selection_list += [ManualBitWidthSelection(filter, bit_width)]

    def get_nodes_to_manipulate_bit_widths(self, graph: Graph):
        """
        Retrieve nodes from the graph that need their bit-widths changed according to the manual bit-width selections.

        Args:
            graph (Graph): The graph containing the nodes to be filtered and manipulated.

        Returns:
            dict: A dictionary mapping nodes to their new bit-widths.
        """
        nodes_to_change_bit_width = {}
        for manual_bit_width_selection in self.manual_bit_width_selection_list:
            filtered_nodes = graph.filter(manual_bit_width_selection.filter)
            nodes_to_change_bit_width.update({n: manual_bit_width_selection.bit_width for n in filtered_nodes})
        return nodes_to_change_bit_width
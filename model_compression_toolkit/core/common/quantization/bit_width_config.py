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
from typing import List, Union, Dict

from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.matchers.node_matcher import BaseNodeMatcher
from model_compression_toolkit.logger import Logger


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
        manual_activation_bit_width_selection_list (List[ManualBitWidthSelection]): A list of ManualBitWidthSelection objects defining manual bit-width configurations.
    """
    def __init__(self,
                 manual_activation_bit_width_selection_list: List[ManualBitWidthSelection] = None):
        self.manual_activation_bit_width_selection_list = [] if manual_activation_bit_width_selection_list is None else manual_activation_bit_width_selection_list

    def __repr__(self):
        # Used for debugging, thus no cover.
        return str(self.__dict__)  # pragma: no cover

    def set_manual_activation_bit_width(self,
                                        filters: Union[List[BaseNodeMatcher], BaseNodeMatcher],
                                        bit_widths: Union[List[int], int]):
        """
        Add a manual bit-width selection to the configuration.

        Args:
            filter (Union[List[BaseNodeMatcher], BaseNodeMatcher]): The filters used to select nodes for bit-width manipulation.
            bit_width (Union[List[int], int]): The bit widths to be applied to the selected nodes.
            If a single value is given it will be applied to all the filters
        """
        filters = [filters] if not isinstance(filters, list) else filters
        bit_widths = [bit_widths] if not isinstance(bit_widths, list) else bit_widths
        if len(bit_widths) > 1 and len(bit_widths) != len(filters):
            Logger.critical(f"Configuration Error: The number of provided bit_width values {len(bit_widths)} "
                            f"must match the number of filters {len(filters)}, or a single bit_width value "
                            f"should be provided for all filters.")
        elif len(bit_widths) == 1 and len(filters) > 1:
            bit_widths = [bit_widths[0] for f in filters]
        for bit_width, filter in zip (bit_widths, filters):
            self.manual_activation_bit_width_selection_list += [ManualBitWidthSelection(filter, bit_width)]

    def get_nodes_to_manipulate_bit_widths(self, graph: Graph) -> Dict:
        """
        Retrieve nodes from the graph that need their bit-widths changed according to the manual bit-width selections.

        Args:
            graph (Graph): The graph containing the nodes to be filtered and manipulated.

        Returns:
            Dict: A dictionary mapping nodes to their new bit-widths.
        """
        nodes_to_change_bit_width = {}
        for manual_bit_width_selection in self.manual_activation_bit_width_selection_list:
            filtered_nodes = graph.filter(manual_bit_width_selection.filter)
            if len(filtered_nodes) == 0:
                Logger.critical(f"Node Filtering Error: No nodes found in the graph for filter {manual_bit_width_selection.filter.__dict__} "
                                f"to change their bit width to {manual_bit_width_selection.bit_width}.")
            for n in filtered_nodes:
                # check if a manual configuration exists for this node
                if n in nodes_to_change_bit_width:
                    Logger.info(
                        f"Node {n} has an existing manual bit width configuration of {nodes_to_change_bit_width.get(n)}. A new manual configuration request of {manual_bit_width_selection.bit_width} has been received, and the previous value is being overridden.")
                nodes_to_change_bit_width.update({n: manual_bit_width_selection.bit_width})
        return nodes_to_change_bit_width
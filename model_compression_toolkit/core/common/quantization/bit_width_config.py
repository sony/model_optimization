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
from dataclasses import dataclass, field
from typing import List, Union, Dict

from model_compression_toolkit.constants import WEIGHTS_ATTRIBUTE, ACTIVATION_ATTRIBUTE
from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.matchers.node_matcher import BaseNodeMatcher
from model_compression_toolkit.logger import Logger

from model_compression_toolkit.core.common.graph.base_node import WeightAttrT

@dataclass
class ManualBitWidthSelection:
    """
   Class to encapsulate the manual bit width selection configuration for a specific filter.

   Attributes:
       filter (BaseNodeMatcher): The filter used to select nodes for bit width manipulation.
       bit_width (int): The bit width to be applied to the selected nodes.
   """
    filter: BaseNodeMatcher
    bit_width: int

@dataclass
class ManualWeightsBitWidthSelection(ManualBitWidthSelection):
    """
   Class to encapsulate the manual weights bit width selection configuration for a specific filter.

   Attributes:
       filter (BaseNodeMatcher): The filter used to select nodes for bit width manipulation.
       bit_width (int): The bit width to be applied to the selected nodes.
       attr (str): The filtered node's attributes to apply bit-width manipulation to.
   """
    attr: WeightAttrT

@dataclass
class BitWidthConfig:
    """
    Class to manage manual bit-width configurations.

    Attributes:
        manual_activation_bit_width_selection_list (List[ManualBitWidthSelection]): A list of ManualBitWidthSelection objects for activation defining manual bit-width configurations.
        manual_activation_bit_width_selection_list (List[ManualWeightsBitWidthSelection]): A list of ManualWeightsBitWidthSelection for weights objects defining manual bit-width configurations.
    """
    manual_activation_bit_width_selection_list: List[ManualBitWidthSelection] = field(default_factory=list)
    manual_weights_bit_width_selection_list: List[ManualWeightsBitWidthSelection] = field(default_factory=list)

    def set_manual_activation_bit_width(self,
                                        filters: Union[List[BaseNodeMatcher], BaseNodeMatcher],
                                        bit_widths: Union[List[int], int]):
        """
        Add a manual bit-width selection for activation to the configuration.

        Args:
            filters (Union[List[BaseNodeMatcher], BaseNodeMatcher]): The filters used to select nodes for bit-width manipulation.
            bit_widths (Union[List[int], int]): The bit widths to be applied to the selected nodes.
            If a single value is given it will be applied to all the filters
        """
        _, bit_widths, filters = self._expand_to_list_filter_and_bit_width(filters, bit_widths)
        for bit_width, filter in zip (bit_widths, filters):
            self.manual_activation_bit_width_selection_list += [ManualBitWidthSelection(filter, bit_width)]

    def set_manual_weights_bit_width(self,
                                        filters: Union[List[BaseNodeMatcher], BaseNodeMatcher],
                                        bit_widths: Union[List[int], int],
                                        attrs: Union[List[str], str]
                                     ):
        """
        Add a manual bit-width selection for weights to the configuration.

        Args:
            filters (Union[List[BaseNodeMatcher], BaseNodeMatcher]): The filters used to select nodes for bit-width manipulation.
            bit_widths (Union[List[int], int]): The bit widths for specified by attrs to be applied to the selected nodes.
            attrs (Union[List[str], str]): The filtered node's attributes to apply bit-width manipulation to.
            If a single value is given it will be applied to all the filters
        """
        attrs, bit_widths, filters = self._expand_to_list_filter_and_bit_width(filters, bit_widths, attrs)
        for attr, bit_width, filter in zip (attrs, bit_widths, filters):
            self.manual_weights_bit_width_selection_list += [ManualWeightsBitWidthSelection(filter, bit_width, attr)]

    def get_nodes_to_manipulate_activation_bit_widths(self, graph: Graph) -> Dict:
        """
        Retrieve nodes from the graph that need their bit-widths changed according to the manual bit-width selections.

        Args:
            graph (Graph): The graph containing the nodes to be filtered and manipulated.

        Returns:
            Dict: A dictionary mapping nodes to their new bit-widths.
        """
        activation_nodes_to_change_bit_width = self._construct_node_to_new_bit_mapping(graph, self.manual_activation_bit_width_selection_list)

        return activation_nodes_to_change_bit_width

    def get_nodes_to_manipulate_weights_bit_widths(self, graph: Graph) -> Dict:
        """
        Retrieve nodes from the graph that need their bit-widths changed according to the manual bit-width selections.

        Args:
            graph (Graph): The graph containing the nodes to be filtered and manipulated.

        Returns:
            Dict: A dictionary mapping nodes to their new bit-widths.
        """
        weights_nodes_to_change_bit_width = self._construct_node_to_new_bit_mapping(graph, self.manual_weights_bit_width_selection_list)

        return weights_nodes_to_change_bit_width



    def _expand_to_list_core(
            self,
            filters: Union[List[BaseNodeMatcher]],
            vals: Union[List[any], any]):
        vals = [vals] if not isinstance(vals, list) else vals
        if len(vals) > 1 and len(vals) != len(filters):
            Logger.critical(f"Configuration Error: The number of provided bit_width values {len(vals)} "
                            f"must match the number of filters {len(filters)}, or a single bit_width value "
                            f"should be provided for all filters.")
        elif len(vals) == 1 and len(filters) > 1:
            vals = [vals[0] for f in filters]
        return vals

    def _expand_to_list_filter_and_bit_width(
            self,
            filters: Union[List[BaseNodeMatcher]],
            bit_widths: Union[List[int], int],
            attrs: Union[List[str], str] = None):
        filters = [filters] if not isinstance(filters, list) else filters
        bit_widths = self._expand_to_list_core(filters, bit_widths)
        attrs = self._expand_to_list_core(filters, attrs)

        return attrs, bit_widths, filters

    def _construct_node_to_new_bit_mapping(self, graph, manual_bit_width_selection_list):
        unit_nodes_to_change_bit_width = {}
        for manual_bit_width_selection in manual_bit_width_selection_list:
            filtered_nodes = graph.filter(manual_bit_width_selection.filter)
            if len(filtered_nodes) == 0:
                Logger.critical(
                    f"Node Filtering Error: No nodes found in the graph for filter {manual_bit_width_selection.filter.__dict__} "
                    f"to change their bit width to {manual_bit_width_selection.bit_width}.")
            for n in filtered_nodes:
                if n.get_node_weights_attributes() is False:
                    Logger.critical(f'The requested attribute to change the bit width for {n} is not existing.')
                # check if a manual configuration exists for this node
                if n in unit_nodes_to_change_bit_width:
                    Logger.info(
                        f"Node {n} has an existing manual bit width configuration of {unit_nodes_to_change_bit_width.get(n)}. A new manual configuration request of {manual_bit_width_selection.bit_width} has been received, and the previous value is being overridden.")
                if isinstance(manual_bit_width_selection_list,  ManualBitWidthSelection):
                    unit_nodes_to_change_bit_width.update({n: manual_bit_width_selection.bit_width})
                elif isinstance(manual_bit_width_selection_list, ManualWeightsBitWidthSelection):
                    unit_nodes_to_change_bit_width.update({n: [manual_bit_width_selection.bit_width, manual_bit_width_selection.attr]})

        return unit_nodes_to_change_bit_width
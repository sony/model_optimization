# Copyright 2021 Sony Semiconductor Israel, Inc. All rights reserved.
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


import numpy as np
from typing import List

from networkx.algorithms.dag import topological_sort
from model_compression_toolkit.core import FrameworkInfo
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.graph.base_graph import Graph
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.core.common.model_builder_mode import ModelBuilderMode
from model_compression_toolkit.core.common.collectors.statistics_collector import BaseStatsCollector


def create_stats_collector_for_node(node: common.BaseNode,
                                    fw_info: FrameworkInfo) -> BaseStatsCollector:
    """
    Gets a node and a groups list and create and return a statistics collector for a node
    according to whether its statistics should be collected and the prior information we
    have about this node.

    Args:
        node: Node to create its statistics collector.
        fw_info: Information relevant to a specific framework about what is out channel axis (for statistics per-channel).

    Returns:
        Statistics collector for statistics collection for the node.
    """

    if node.is_activation_quantization_enabled():
        min_output = getattr(node.prior_info, 'min_output', None)
        max_output = getattr(node.prior_info, 'max_output', None)
        stats_collector = common.StatsCollector(out_channel_axis=fw_info.out_channel_axis_mapping.get(node.type),
                                                init_min_value=min_output,
                                                init_max_value=max_output)
    else:
        stats_collector = common.NoStatsCollector()

    return stats_collector


def create_tensor2node(graph: common.Graph,
                       node: common.BaseNode,
                       fw_info: common.FrameworkInfo):
    """
    Force statistic collector creation and assignment for a node.
    Args:
        graph: Graph of the node (for retrieving the current tensor).
        node: Node to create a tensor for.
        fw_info: Specific framework information (for example, output channels index).

    """
    current_sc = graph.get_out_stats_collector(node)
    is_list_nostat_collectors = isinstance(current_sc, list) and len([sc for sc in current_sc if not isinstance(sc, common.NoStatsCollector)]) == 0
    if isinstance(current_sc, common.NoStatsCollector) or current_sc is None or is_list_nostat_collectors:
        stats_collector = common.StatsCollector(fw_info.out_channel_axis_mapping.get(node.type))
        graph.set_out_stats_collector_to_node(node, stats_collector)


class ModelCollector:
    """
    Build a model from a graph for statistics collection purposes.
    The ModelCollector builds a float model that its outputs are all layers outputs, so after
    inferring using this model, statistics of output layers can be gathered and be used
    for thresholds calculations.
    """

    def __init__(self, graph: Graph,
                 fw_impl: FrameworkImplementation,
                 fw_info: FrameworkInfo,
                 qc: common.QuantizationConfig = common.DEFAULTCONFIG):
        """
        Build a model from a graph per framework for statistics collection.

        Args:
            graph: Graph to build a model from it.
            fw_impl: FrameworkImplementation object with a specific framework methods implementation.
            fw_info: FrameworkInfo object with a specific framework information.
            qc: Quantization configuration containing parameters for how the graph should be quantized.
        """

        self.fw_impl = fw_impl
        self.fw_info = fw_info

        # Assign statisitcs collectors to nodes
        for n in graph.get_topo_sorted_nodes():
            sc = create_stats_collector_for_node(n, fw_info=fw_info)  # Get static collector for the node
            # If we use bias correction, and the node has kernel weights to quantize, we need to make sure
            # its previous nodes' tensors are consistent with this node.
            kernel_attr = fw_info.get_kernel_op_attributes(n.type)[0]
            if qc.weights_bias_correction and kernel_attr is not None and n.is_weights_quantization_enabled(kernel_attr):
                for ie in graph.incoming_edges(n):
                    input_node = ie.source_node
                    create_tensor2node(graph,
                                       input_node,
                                       fw_info)
            if sc is not None:
                graph.set_out_stats_collector_to_node(n, sc)


        outputs_nodes = []  # List of graph nodes, the model should output their outputs.
        self.stats_containers_list = []  # List of output statistics containers of nodes ordered
        # the same as outputs_nodes so statistics of outputs can be gathered for the correct statistics container.
        for n in graph.nodes():
            out_stats_container = graph.get_out_stats_collector(n)
            if isinstance(out_stats_container, list):  # If layer has multiple outputs
                # Append nodes to output and track their statistics only if
                # they actually collect statistics.
                if len([x for x in out_stats_container if not isinstance(x, common.NoStatsCollector)]) > 0:
                    mark2out = True
                    for sc in out_stats_container:
                        # Output only if statistics should be gathered
                        if sc.require_collection() and mark2out:
                            mark2out = False
                            outputs_nodes.append(n)  # Append node several times (as number of outputs it has)
                    self.stats_containers_list.append(out_stats_container)

            else:  # A single output
                if out_stats_container.require_collection():
                    outputs_nodes.append(n)
                    self.stats_containers_list.append(out_stats_container)

        # Build a float model and output all layers' outputs
        # (that should be collected) as the model's outputs
        self.model, _ = self.fw_impl.model_builder(graph,
                                                   mode=ModelBuilderMode.FLOAT,
                                                   append2output=outputs_nodes,
                                                   fw_info=self.fw_info)

    def infer(self, inputs_list: List[np.ndarray]):
        """
        Pass inputs through the model of the ModelCollector,
        and update statistics in all statistics containers the ModelCollector holds.

        Args:
            inputs_list: Inputs for the model inferring.

        """

        # TODO: Thinking about delegating collections to framework
        # TODO: migrate datasets to framework datasets
        tensor_data = self.fw_impl.run_model_inference(self.model, inputs_list)
        for td, sc in zip(tensor_data, self.stats_containers_list):
            if isinstance(sc, (list, tuple)):
                if not isinstance(td, (list, tuple)):
                    Logger.critical('\'tensor_data\' must be a list or a tuple if \'stats_containers_list\' contains lists or tuples.') # pragma: no cover
                if len(sc) != len(td):
                    Logger.critical('\'tensor_data\' and \'stats_containers_list\' must have matching lengths') # pragma: no cover
                for tdi, sci in zip(td, sc):
                    sci.update_statistics(self.fw_impl.to_numpy(tdi))
            else:
                sc.update_statistics(self.fw_impl.to_numpy(td))

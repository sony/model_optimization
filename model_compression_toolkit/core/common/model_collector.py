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

from model_compression_toolkit import FrameworkInfo
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.graph.base_graph import Graph
from model_compression_toolkit.core.common.logger import Logger
from model_compression_toolkit.core.common.model_builder_mode import ModelBuilderMode


class ModelCollector:
    """
    Build a Keras model from a graph for statistics collection purposes.
    The ModelCollector builds a float model that its outputs are all layers outputs, so after
    inferring using this model, statistics of output layers can be gathered and be used
    for thresholds calculations.
    """

    def __init__(self, graph: Graph, fw_impl: FrameworkImplementation, fw_info: FrameworkInfo):
        """
        Build a Keras model from the passed graph, and set the model's
        outputs to be all layers' outputs.

        Args:
            graph: Graph to build a model from it.
            fw_impl: FrameworkImplementation object with a specific framework methods implementation.

        """

        self.graph = graph
        self.fw_impl = fw_impl
        self.fw_info = fw_info

        node2fetch = []  # List of graph nodes, the model should output their outputs.
        stats_containers_list = []  # List of output statistics containers of nodes ordered
        # the same as node2fetch so statistics of outputs can be gathered for the correct statistics container.

        for n in self.graph.nodes():
            out_stats_container = self.graph.get_out_stats_collector(n)
            if isinstance(out_stats_container, list):  # If layer has multiple outputs
                # Append nodes to output and track their statistics only if
                # they actually collect statistics.
                if len([x for x in out_stats_container if not isinstance(x, common.NoStatsCollector)]) > 0:
                    mark2fetch = True
                    for sc in out_stats_container:
                        # Output only if statistics should be gathered
                        if sc.require_collection() and mark2fetch:
                            mark2fetch = False
                            node2fetch.append(n)  # Append node several times (as number of outputs it has)
                    stats_containers_list.append(out_stats_container)

            else:  # A single output
                if out_stats_container.require_collection():
                    node2fetch.append(n)
                    stats_containers_list.append(out_stats_container)

        self.stats_containers_list = stats_containers_list

        # Build a float model and output all layers' outputs
        # (that should be collected) as the model's outputs
        self.model, _ = self.fw_impl.model_builder(self.graph,
                                                   mode=ModelBuilderMode.FLOAT,
                                                   append2output=node2fetch,
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
                    Logger.exception(
                        '"tensor_data" must be a list or a tuple if the model tensor_list is a list or a tuple')
                if len(sc) != len(td):
                    Logger.exception('"tensor_data" and the model tensor_list must be of the same length')
                for tdi, sci in zip(td, sc):
                    sci.update_statistics(self.fw_impl.to_numpy(tdi))
            else:
                sc.update_statistics(self.fw_impl.to_numpy(td))

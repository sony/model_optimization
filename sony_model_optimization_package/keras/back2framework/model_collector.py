# ===============================================================================
# Copyright (c) 2021, Sony Semiconductors Israel, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ===============================================================================


from typing import List

import numpy as np

from sony_model_optimization_package.common.graph.base_graph import Graph
from sony_model_optimization_package.common.logger import Logger
from sony_model_optimization_package.keras.back2framework.model_builder import model_builder, ModelBuilderMode


class ModelCollector(object):
    """
    Build a Keras model from a graph for statistics collection purposes.
    The ModelCollector builds a float model that its outputs are all layers outputs, so after
    inferring using this model, statistics of output layers can be gathered and be used
    for thresholds calculations.
    """

    def __init__(self, graph: Graph):
        """
        Build a Keras model from the passed graph, and set the model's
        outputs to be all layers' outputs.

        Args:
            graph: Graph to build a model from it.
        """

        self.graph = graph
        node2fetch = []  # List of graph nodes, the model should output their outputs.
        stats_containers_list = []  # List of output statistics containers of nodes ordered
        # the same as node2fetch so statistics of outputs can be gathered for the correct statistics container.

        for n in self.graph.nodes():
            out_stats_container = self.graph.get_out_stats_collector(n)

            if isinstance(out_stats_container, list):  # If layer has multiple outputs
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
        self.model, _ = model_builder(self.graph,
                                      mode=ModelBuilderMode.FLOAT,
                                      append2output=node2fetch)

    def infer(self, inputs_list: List[np.ndarray]):
        """
        Pass inputs through the model of the ModelCollector,
        and update statistics in all statistics containers the ModelCollector holds.

        Args:
            inputs_list: Inputs for the model inferring.
        """
        # TODO: Thinking about delegating collections to framework
        # TODO: migrate datasets to framework datasets
        tensor_data = self.model(list(inputs_list))
        for td, sc in zip(tensor_data, self.stats_containers_list):
            if isinstance(sc, (list, tuple)):
                if not isinstance(td, (list, tuple)):
                    Logger.exception(
                        '"tensor_data" must be a list or a tuple if the model tensor_list is a list or a tuple')
                if len(sc) != len(td):
                    Logger.exception('"tensor_data" and the model tensor_list must be of the same length')
                for tdi, sci in zip(td, sc):
                    sci.update_statistics(tdi.numpy())
            else:
                sc.update_statistics(td.numpy())

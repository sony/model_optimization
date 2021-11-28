# Copyright 2021 Sony Semiconductors Israel, Inc. All rights reserved.
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

from typing import Callable
from typing import Dict, List

from model_compression_toolkit.common.graph.base_graph import Graph
from model_compression_toolkit.common.mixed_precision.kpi import KPI
from model_compression_toolkit.common.mixed_precision.mixed_precision_quantization_config import \
    MixedPrecisionQuantizationConfig
from model_compression_toolkit.common.framework_info import FrameworkInfo


class MixedPrecisionSearchManager(object):
    """
    Class to wrap and manage the search process of a mixed-precision configuration.
    """

    def __init__(self,
                 graph: Graph,
                 qc: MixedPrecisionQuantizationConfig,
                 fw_info: FrameworkInfo,
                 get_sensitivity_evaluation: Callable):
        """

        Args:
            graph: Graph to search for its MP configuration.
            qc: Quantization configuration for how the graph should be quantized.
            fw_info: FrameworkInfo object about the specific framework (e.g., attributes of different layers' weights to quantize).
            get_sensitivity_evaluation: Framework specific function to retrieve a metric computation function.
        """

        self.graph = graph
        self.qc = qc
        self.fw_info = fw_info
        self.get_sensitivity_evaluation = get_sensitivity_evaluation
        self.metrics_weights = self.qc.distance_weighting_method
        self.layer_to_bitwidth_mapping = self.get_search_space()
        self.compute_metric_fn = self.get_sensitivity_metric()

    def get_search_space(self) -> Dict[int, List[int]]:
        """
        The search space is a mapping from a node's index to a list of integers (possible bitwidths for
        the node).

        Returns:
            The entire search space of the graph.
        """

        indices_mapping = {}
        nodes_to_configure = self.graph.get_configurable_sorted_nodes()
        for idx, n in enumerate(nodes_to_configure):
            # For each node, get all possible bitwidth indices for it
            # (which is a list from 0 to the length of the candidates qc list of the node).
            indices_mapping[idx] = list(range(len(n.candidates_weights_quantization_cfg)))  # all search_methods space
        return indices_mapping


    def get_sensitivity_metric(self) -> Callable:
        """

        Returns: Return a function (from the framework implementation) to compute a metric that
        indicates the similarity of the mixed-precision model (to the float model) for a given
        mixed-precision configuration.

        """
        # Get from the framework an evaluation function on how a MP configuration,
        # affects the expected loss.
        compute_metric_fn = self.get_sensitivity_evaluation(self.graph,
                                                            self.qc,
                                                            self.metrics_weights)
        return compute_metric_fn

    def get_kpi_metric(self) -> Callable:
        """

        Returns: A function to compute the KPI of a graph for a mixed-precision bitwidth
        configuration.

        """

        def _compute_kpi(mp_model_config: List[int]) -> KPI:
            """
            Compute and return the KPI of a graph for a given mixed-precision bitwidth
            configuration.

            Args:
                mp_model_config: Mixed-precision bitwidth configuration (list of integers).

            Returns:
                KPI of a model when using the passed mixed-precision configuration.

            """
            weights_memory = 0

            # Go over all nodes that shold be taken into consideration when computing the KPI.
            mp_nodes = self.graph.get_configurable_sorted_nodes_names()
            for n in self.graph.nodes:
                if n.name in mp_nodes:
                    node_idx = mp_nodes.index(n.name)
                    node_nbits = n.candidates_weights_quantization_cfg[mp_model_config[node_idx]].weights_n_bits
                elif n.candidates_weights_quantization_cfg is not None:
                    # The only valid way to get here is if the node is reused (which means that we're not looking
                    # for its configuration), and we ignore it when computing the KPI (as the base node will acount
                    # for it).
                    assert n.reuse, "If node has candidates it should be part of the configurable nodes," \
                                    " unless it's a reused node"
                    node_nbits = 0  # Ignore reused nodes is the KPI computation.
                else:  # No weights quantization
                    node_nbits = 0
                node_num_params = 0

                # Consider only the weights that should be quantized.
                for attr in self.fw_info.get_kernel_op_attributes(n.layer_class):
                    if attr is not None:
                        node_num_params += n.get_weights_by_keys(attr).flatten().shape[0]

                node_memory_in_bytes = node_num_params * node_nbits / 8.0
                weights_memory += node_memory_in_bytes

            return KPI(weights_memory=weights_memory)

        return _compute_kpi

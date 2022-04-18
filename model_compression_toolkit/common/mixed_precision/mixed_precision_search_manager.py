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
import numpy as np

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
        self.min_activation_cfg = self.get_min_activation_cfg()
        self.min_weights_cfg = self.get_min_weights_cfg()

    def get_search_space(self) -> Dict[int, List[int]]:
        """
        The search space is a mapping from a node's index to a list of integers (possible bitwidths candidates indeces
        for the node).

        Returns:
            The entire search space of the graph.
        """

        indices_mapping = {}
        nodes_to_configure = self.graph.get_configurable_sorted_nodes()
        for idx, n in enumerate(nodes_to_configure):
            # For each node, get all possible bitwidth indices for it
            # (which is a list from 0 to the length of the candidates qc list of the node).
            indices_mapping[idx] = list(range(len(n.candidates_quantization_cfg)))  # all search_methods space
        return indices_mapping

    def get_min_activation_cfg(self):
        """
        Builds a mixed-precision config with the bitwidth indexes for model with minimal activation KPI.

        Returns: A mp configuration (list of indices)

        """
        nodes_to_configure = self.graph.get_configurable_sorted_nodes()
        nodes_activation_bitwidth_candidates = [[c.activation_quantization_cfg.activation_n_bits for c in
                                                 n.candidates_quantization_cfg] for n in nodes_to_configure]
        return [np.argmin(n_candidates) for n_candidates in nodes_activation_bitwidth_candidates]

    def get_min_weights_cfg(self):
        """
        Builds a mixed-precision config with the bitwidth indexes for model with minimal weights KPI.

        Returns: A mp configuration (list of indices)

        """
        nodes_to_configure = self.graph.get_configurable_sorted_nodes()
        nodes_weights_bitwidth_candidates = [[c.weights_quantization_cfg.weights_n_bits for c in
                                              n.candidates_quantization_cfg] for n in nodes_to_configure]
        return [np.argmin(n_candidates) for n_candidates in nodes_weights_bitwidth_candidates]

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

        def _compute_kpi(mp_model_config: List[int],
                         compute_weights_kpi: bool = True,
                         compute_activation_kpi: bool = True) -> KPI:
            """
            Compute and return the KPI of a graph for a given mixed-precision bitwidth
            configuration.

            Args:
                mp_model_config: Mixed-precision bitwidth configuration (list of integers).
                compute_weights_kpi: Flag that specifies to run computation for weights memory.
                compute_activation_kpi: Flag that specifies to run computation for activation memory.

            Returns:
                KPI of a model when using the passed mixed-precision configuration.

            """
            assert compute_weights_kpi or compute_activation_kpi, \
                "Compute KPI need at least one of weights/activation KPI to compute."

            weights_memory = 0
            activations_memory = 0

            # Go over all nodes that should be taken into consideration when computing the KPI.
            mp_nodes = self.graph.get_configurable_sorted_nodes_names()
            for n in self.graph.nodes:
                if n.name in mp_nodes:
                    node_idx = mp_nodes.index(n.name)
                    node_qc = n.candidates_quantization_cfg[mp_model_config[node_idx]]
                    node_nbits = (node_qc.weights_quantization_cfg.weights_n_bits,
                                  node_qc.activation_quantization_cfg.activation_n_bits)
                elif n.is_weights_quantization_enabled() and n.has_weights_to_quantize(self.fw_info):
                    # The only two valid ways to get here are:
                    # 1) If the node is reused (which means that we're not looking for its configuration),
                    #       and we ignore it when computing the KPI (as the base node will account for it).
                    #       In activation KPI calculation -
                    #       if we sum all inputs as a metric then we don't want to skip reused nodes.
                    # 2) If mixed-precision search is only on activation candidates,
                    #       and weights quantization n_bits is fixed.
                    assert n.reuse or n.is_all_weights_candidates_equal(), \
                        "If node has candidates it should be part of the configurable nodes, unless it's a reused " \
                        "node or the candidates only differ in activation bitwidth"
                    node_nbits = (0, 0)  # Ignore reused nodes or weights quantization
                    # only (if no weights mixed-precision) in the KPI computation.
                else:  # No quantization
                    node_nbits = (0, 0)

                # Weights memory size computation
                # Consider only the weights that should be quantized.
                if compute_weights_kpi and n.is_weights_quantization_enabled() and \
                        not n.is_all_weights_candidates_equal():
                    node_num_weights_params = 0
                    for attr in self.fw_info.get_kernel_op_attributes(n.type):
                        if attr is not None:
                            node_num_weights_params += n.get_weights_by_keys(attr).flatten().shape[0]

                    node_weights_memory_in_bytes = node_num_weights_params * node_nbits[0] / 8.0
                    weights_memory += node_weights_memory_in_bytes

                    # Activation memory size computation
                    # Currently, consider layer's activation size as size of layer's output,
                    # and total model activations' size as sum of layers' output.
                if compute_activation_kpi and n.is_activation_quantization_enabled() and \
                        not n.is_all_activation_candidates_equal():
                    node_output_size = n.get_total_output_params()
                    node_activation_memory_in_bytes = node_output_size * node_nbits[1] / 8.0
                    activations_memory += node_activation_memory_in_bytes

            return KPI(weights_memory=weights_memory,
                       activation_memory=activations_memory)

        return _compute_kpi

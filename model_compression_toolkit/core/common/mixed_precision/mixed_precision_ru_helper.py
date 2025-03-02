# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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
from typing import List, Set, Dict, Optional, Tuple, Any

import numpy as np

from model_compression_toolkit.core import FrameworkInfo
from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization import \
    RUTarget
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization_calculator import \
    ResourceUtilizationCalculator, BitwidthMode, TargetInclusionCriterion
from model_compression_toolkit.core.common.quantization.node_quantization_config import NodeWeightsQuantizationConfig, \
    NodeActivationQuantizationConfig


class MixedPrecisionRUHelper:
    """ Helper class for resource utilization computations for mixed precision optimization. """

    def __init__(self, graph: Graph, fw_info: FrameworkInfo, fw_impl: FrameworkImplementation):
        self.graph = graph
        self.fw_info = fw_info
        self.fw_impl = fw_impl
        self.ru_calculator = ResourceUtilizationCalculator(graph, fw_impl, fw_info)

    def compute_utilization(self, ru_targets: Set[RUTarget], mp_cfg: List[int]) -> Dict[RUTarget, np.ndarray]:
        """
        Compute utilization of requested targets for a specific configuration in the format expected by LP problem
        formulation namely a vector of ru values for relevant memory elements (nodes or cuts) in a constant order
        (between calls).

        Args:
            ru_targets: resource utilization targets to compute.
            mp_cfg: a list of candidates indices for configurable layers.

        Returns:
            Dict of the computed utilization per target.
        """

        ru = {}
        act_qcs, w_qcs = self.get_quantization_candidates(mp_cfg)
        if RUTarget.WEIGHTS in ru_targets:
            wu = self._weights_utilization(w_qcs)
            ru[RUTarget.WEIGHTS] = np.array(list(wu.values()))

        if RUTarget.ACTIVATION in ru_targets:
            au = self._activation_utilization(act_qcs)
            ru[RUTarget.ACTIVATION] = np.array(list(au.values()))

        if RUTarget.BOPS in ru_targets:
            ru[RUTarget.BOPS] = self._bops_utilization(act_qcs=act_qcs, w_qcs=w_qcs)

        if RUTarget.TOTAL in ru_targets:
            raise ValueError('Total target should be computed based on weights and activations targets.')

        assert len(ru) == len(ru_targets), (f'Mismatch between the number of computed and requested metrics.'
                                            f'Requested {ru_targets}')
        return ru

    def get_quantization_candidates(self, mp_cfg) \
            -> Tuple[Dict[str, NodeActivationQuantizationConfig], Dict[str, NodeWeightsQuantizationConfig]]:
        """
        Retrieve quantization candidates objects for weights and activations from the configuration list.

        Args:
            mp_cfg: a list of candidates indices for configurable layers.

        Returns:
            A mapping between nodes to weights quantization config, and a mapping between nodes and activation
            quantization config.
        """
        mp_nodes = self.graph.get_configurable_sorted_nodes(self.fw_info)
        node_qcs = {n: n.candidates_quantization_cfg[mp_cfg[i]] for i, n in enumerate(mp_nodes)}
        act_qcs = {n.name: cfg.activation_quantization_cfg for n, cfg in node_qcs.items()}
        w_qcs = {n.name: cfg.weights_quantization_cfg for n, cfg in node_qcs.items()}
        return act_qcs, w_qcs

    def _weights_utilization(self, w_qcs: Dict[str, NodeWeightsQuantizationConfig]) -> Dict[str, float]:
        """
        Compute weights utilization for configurable weights.

        Args:
            w_qcs: nodes quantization configuration to compute.

        Returns:
            Weight utilization per node.
        """
        _, nodes_util, _ = self.ru_calculator.compute_weights_utilization(target_criterion=TargetInclusionCriterion.AnyQuantized,
                                                                          bitwidth_mode=BitwidthMode.QCustom,
                                                                          w_qcs=w_qcs)
        nodes_util = {n: u.bytes for n, u in nodes_util.items()}
        return nodes_util

    def _activation_utilization(self, act_qcs: Dict[str, NodeActivationQuantizationConfig]) -> Dict[Any, float]:
        """
        Compute activation utilization using MaxCut for all quantized nodes.

        Args:
            act_qcs: nodes activation configuration.

        Returns:
            Activation utilization per cut.
        """
        _, cuts_util, *_ = self.ru_calculator.compute_activation_utilization_by_cut(
            TargetInclusionCriterion.AnyQuantized, bitwidth_mode=BitwidthMode.QCustom, act_qcs=act_qcs)
        cuts_util = {c: u.bytes for c, u in cuts_util.items()}
        return cuts_util

    def _bops_utilization(self,
                          act_qcs: Optional[Dict[str, NodeActivationQuantizationConfig]],
                          w_qcs: Optional[Dict[str, NodeWeightsQuantizationConfig]]) -> np.ndarray:
        """
        Computes a resource utilization vector with the respective bit-operations (BOPS) count
        according to the given mixed-precision configuration.

        Args:
            act_qcs: nodes activation configuration.
            w_qcs: nodes quantization configuration to compute.

        Returns:
            A vector of node's BOPS count.
        """
        _, detailed_bops = self.ru_calculator.compute_bops(TargetInclusionCriterion.Any, BitwidthMode.QCustom,
                                                           act_qcs=act_qcs, w_qcs=w_qcs)
        return np.array(list(detailed_bops.values()))

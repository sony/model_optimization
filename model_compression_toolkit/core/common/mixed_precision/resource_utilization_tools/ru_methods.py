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
from typing import List, Set, Dict, Optional, Tuple

import numpy as np

from model_compression_toolkit.core import FrameworkInfo
from model_compression_toolkit.core.common import Graph, BaseNode
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.graph.memory_graph.cut import Cut
from model_compression_toolkit.core.common.graph.virtual_activation_weights_node import VirtualActivationWeightsNode
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization import \
    RUTarget
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization_calculator import \
    ResourceUtilizationCalculator, BitwidthMode, TargetInclusionCriterion
from model_compression_toolkit.core.common.quantization.node_quantization_config import NodeWeightsQuantizationConfig, \
    NodeActivationQuantizationConfig


# TODO take into account Virtual nodes. Are candidates defined with respect to virtual or original nodes?
#  Can we use the virtual graph only for bops and the original graph for everything else?

class MixedPrecisionRUHelper:
    """ Helper class for resource utilization computations for mixed precision optimization. """

    def __init__(self, graph: Graph, fw_info: FrameworkInfo, fw_impl: FrameworkImplementation):
        self.graph = graph
        self.fw_info = fw_info
        self.fw_impl = fw_impl
        self.ru_calculator = ResourceUtilizationCalculator(graph, fw_impl, fw_info)

    def compute_utilization(self, ru_targets: Set[RUTarget], mp_cfg: Optional[List[int]]) -> Dict[RUTarget, np.ndarray]:
        """
        Compute utilization of requested targets for a specific configuration in the format expected by LP problem
        formulation, namely an array of ru values corresponding to graph's configurable nodes in the topological order.
        For activation target, the array contains values for activation cuts in unspecified order (as long as it is
        consistent between configurations).

        Args:
            ru_targets: resource utilization targets to compute.
            mp_cfg: a list of candidates indices for configurable layers.

        Returns:
            Dict of the computed utilization per target.
        """

        ru = {}

        act_qcs, w_qcs = self.get_configurable_qcs(mp_cfg) if mp_cfg else (None, None)
        w_util = None
        if RUTarget.WEIGHTS in ru_targets:
            w_util = self._weights_utilization(w_qcs)
            ru[RUTarget.WEIGHTS] = np.array(list(w_util.values()))

        # TODO make mp agnostic to activation method
        if RUTarget.ACTIVATION in ru_targets:
            act_util = self._activation_maxcut_utilization(act_qcs)
            ru[RUTarget.ACTIVATION] = np.array(list(act_util.values()))

        # TODO use maxcut
        if RUTarget.TOTAL in ru_targets:
            act_tensors_util = self._activation_tensor_utilization(act_qcs)
            w_util = w_util or self._weights_utilization(w_qcs)
            total = {n: (w_util.get(n, 0), act_tensors_util.get(n, 0))
                     # for n in self.graph.nodes if n in act_tensors_util or n in w_util}
                     for n in self.graph.get_topo_sorted_nodes() if n in act_tensors_util or n in w_util}
            ru[RUTarget.TOTAL] = np.array(list(total.values()))

        if RUTarget.BOPS in ru_targets:
            ru[RUTarget.BOPS] = self._bops_utilization(mp_cfg)

        return ru

    def get_configurable_qcs(self, mp_cfg) \
            -> Tuple[Dict[BaseNode, NodeActivationQuantizationConfig], Dict[BaseNode, NodeWeightsQuantizationConfig]]:
        """
        Retrieve quantization candidates objects for weights and activations from the configuration list.

        Args:
            mp_cfg: a list of candidates indices for configurable layers.

        Returns:
            Mapping between nodes to weights quantization config, and a mapping between nodes and activation
            quantization config.
        """
        mp_nodes = self.graph.get_configurable_sorted_nodes(self.fw_info)
        node_qcs = {n: n.candidates_quantization_cfg[mp_cfg[i]] for i, n in enumerate(mp_nodes)}
        act_qcs = {n: node_qcs[n].activation_quantization_cfg
                   for n in self.graph.get_activation_configurable_nodes()}
        w_qcs = {n: node_qcs[n].weights_quantization_cfg
                 for n in self.graph.get_weights_configurable_nodes(self.fw_info)}
        return act_qcs, w_qcs

    def _weights_utilization(self, w_qcs: Optional[Dict[BaseNode, NodeWeightsQuantizationConfig]]) -> Dict[BaseNode, float]:
        """
        Compute weights utilization for configurable weights if configuration is passed,
        or for non-configurable nodes otherwise.

        Args:
            w_qcs: nodes quantization configuration to compute, or None.

        Returns:
            Weight utilization per node.
        """
        if w_qcs:
            target_criterion = TargetInclusionCriterion.QConfigurable
            bitwidth_mode = BitwidthMode.QCustom
        else:
            target_criterion = TargetInclusionCriterion.QNonConfigurable
            bitwidth_mode = BitwidthMode.QDefaultSP

        _, nodes_util, _ = self.ru_calculator.compute_weights_utilization(target_criterion=target_criterion,
                                                                          bitwidth_mode=bitwidth_mode,
                                                                          w_qcs=w_qcs)
        nodes_util = {n: u.bytes for n, u in nodes_util.items()}
        return nodes_util

    def _activation_maxcut_utilization(self, act_qcs: Optional[Dict[BaseNode, NodeActivationQuantizationConfig]]) \
            -> Optional[Dict[Cut, float]]:
        """
        Compute activation utilization using MaxCut for all quantized nodes if configuration is passed.

        Args:
            act_qcs: nodes activation configuration or None.

        Returns:
            Activation utilization per cut, or empty dict if no configuration was passed.
        """
        if act_qcs:
            _, cuts_util, _ = self.ru_calculator.compute_cut_activation_utilization(TargetInclusionCriterion.AnyQuantized,
                                                                                    bitwidth_mode=BitwidthMode.QCustom,
                                                                                    act_qcs=act_qcs)
            cuts_util = {c: u.bytes for c, u in cuts_util.items()}
            return cuts_util

        # Computing non-configurable nodes resource utilization for max-cut is included in the calculation of the
        # configurable nodes.
        return {}

    def _activation_tensor_utilization(self, act_qcs: Optional[Dict[BaseNode, NodeActivationQuantizationConfig]]):
        """
        Compute activation tensors utilization fo configurable nodes if configuration is passed or
        for non-configurable nodes otherwise.

        Args:
            act_qcs: activation quantization configuration or None.

        Returns:
            Activation utilization per node.
        """
        if act_qcs:
            target_criterion = TargetInclusionCriterion.QConfigurable
            bitwidth_mode = BitwidthMode.QCustom
        else:
            target_criterion = TargetInclusionCriterion.QNonConfigurable
            bitwidth_mode = BitwidthMode.QDefaultSP

        _, nodes_util = self.ru_calculator.compute_activation_tensors_utilization(target_criterion=target_criterion,
                                                                                  bitwidth_mode=bitwidth_mode,
                                                                                  act_qcs=act_qcs)
        return {n: u.bytes for n, u in nodes_util.items()}

    def _bops_utilization(self, mp_cfg: List[int]):
        """
        Computes a resource utilization vector with the respective bit-operations (BOPS) count for each configurable node,
        according to the given mixed-precision configuration of a virtual graph with composed nodes.

        Args:
            mp_cfg: A mixed-precision configuration (list of candidates index for each configurable node)

        Returns: A vector of node's BOPS count.
        Note that the vector is not necessarily of the same length as the given config.

        """
        # TODO keeping old implementation for now

        # BOPs utilization method considers non-configurable nodes, therefore, it doesn't need separate implementation
        # for non-configurable nodes for setting a constraint (no need for separate implementation for len(mp_cfg) = 0).

        virtual_bops_nodes = [n for n in self.graph.get_topo_sorted_nodes() if isinstance(n, VirtualActivationWeightsNode)]

        mp_nodes = self.graph.get_configurable_sorted_nodes_names(self.fw_info)

        bops = [n.get_bops_count(self.fw_impl, self.fw_info, candidate_idx=_get_node_cfg_idx(n, mp_cfg, mp_nodes))
                for n in virtual_bops_nodes]

        return np.array(bops)


def _get_node_cfg_idx(node: BaseNode, mp_cfg: List[int], sorted_configurable_nodes_names: List[str]) -> int:
    """
    Returns the index of a node's quantization configuration candidate according to the given
    mixed-precision configuration. If the node is not configurable, then it must have a single configuration,
    therefore, the index 0 is returned.

    Args:
        node: A node to get its candidate configuration index.
        mp_cfg: A mixed-precision configuration (list of candidates index for each configurable node)
        sorted_configurable_nodes_names: A list of configurable nodes names.

    Returns: An index (integer) of a node's quantization configuration candidate.
    """

    if node.name in sorted_configurable_nodes_names:
        node_idx = sorted_configurable_nodes_names.index(node.name)
        return mp_cfg[node_idx]
    else:    # pragma: no cover
        assert len(node.candidates_quantization_cfg) > 0, \
            "Any node should have at least one candidate configuration."
        return 0

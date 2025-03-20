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
from typing import List, Set, Dict, Tuple

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
        Compute utilization of requested targets for a specific configuration:
          for weights and bops - total utilization,
          for activations and total - utilization per cut.

        Args:
            ru_targets: resource utilization targets to compute.
            mp_cfg: a list of candidates indices for configurable layers.

        Returns:
            Dict of the computed utilization per target, as 1d vector.
        """
        act_qcs, w_qcs = self.get_quantization_candidates(mp_cfg)

        ru, detailed_ru = self.ru_calculator.compute_resource_utilization(TargetInclusionCriterion.AnyQuantized,
                                                                          BitwidthMode.QCustom,
                                                                          act_qcs=act_qcs,
                                                                          w_qcs=w_qcs,
                                                                          ru_targets=ru_targets,
                                                                          allow_unused_qcs=True,
                                                                          return_detailed=True)

        ru_dict = {k: np.array([v]) for k, v in ru.get_resource_utilization_dict(restricted_only=True).items()}
        # For activation and total we need utilization per cut, as different mp configurations might result in
        # different cuts to be maximal.
        for target in [RUTarget.ACTIVATION, RUTarget.TOTAL]:
            if target in ru_dict:
                ru_dict[target] = np.array(list(detailed_ru[target].values()))

        assert all(v.ndim == 1 for v in ru_dict.values())
        if RUTarget.ACTIVATION in ru_targets and RUTarget.TOTAL in ru_targets:
            assert ru_dict[RUTarget.ACTIVATION].shape == ru_dict[RUTarget.TOTAL].shape

        assert len(ru_dict) == len(ru_targets), (f'Mismatch between the number of computed and requested metrics.'
                                                 f'Requested {ru_targets}')
        return ru_dict

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

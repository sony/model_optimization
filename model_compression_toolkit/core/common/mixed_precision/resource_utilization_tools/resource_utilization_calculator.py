# Copyright 2025 Sony Semiconductor Israel, Inc. All rights reserved.
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
from collections import defaultdict
from copy import deepcopy
from enum import Enum, auto
from typing import Dict, NamedTuple, Optional, Tuple, List, Iterable, Union, Literal, Sequence, Set

from model_compression_toolkit.logger import Logger
from model_compression_toolkit.constants import FLOAT_BITWIDTH
from model_compression_toolkit.core import FrameworkInfo
from model_compression_toolkit.core.common import Graph, BaseNode
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.graph.edge import EDGE_SINK_INDEX
from model_compression_toolkit.core.common.graph.memory_graph.compute_graph_max_cut import compute_graph_max_cut
from model_compression_toolkit.core.common.graph.memory_graph.cut import Cut
from model_compression_toolkit.core.common.graph.memory_graph.memory_graph import MemoryGraph
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization import \
    RUTarget, ResourceUtilization
from model_compression_toolkit.core.common.quantization.node_quantization_config import NodeWeightsQuantizationConfig, \
    NodeActivationQuantizationConfig


class BitwidthMode(Enum):
    """
    Bit-width configuration for resource utilization computation.

    Float: original un-quantized configuration. Assumed to be 32-bit float.
    QMaxBit: maximal bit-width configurations. Assigns each node its maximal available precision according to the
      target platform capabilities.
    QMinBit: minimal bit-width configuration. Assigns each node its minimal available precision according to the
      target platform capabilities.
    QCustom: explicitly provided bit-width configuration.
    QDefaultSP: default single-precision bit-width configuration. Can be used either in a single-precision mode,
      or along with TargetInclusionCriterion.QNonConfigurable, which computes the resource utilization only for
      single-precision nodes. To compute custom single precision configuration, use QCustom.
    """
    Float = auto()
    Q8Bit = auto()
    QMaxBit = auto()
    QMinBit = auto()
    QCustom = auto()
    QDefaultSP = auto()


class TargetInclusionCriterion(Enum):
    """
    Target nodes / parameters to include for resource utilization computation.

    QConfigurable: configurable for Mixed Precision targets (multiple quantization candidates).
    QNonConfigurable: non-configurable targets (single quantization candidate).
    AnyQuantized: any quantized targets (configurable and non-configurable).
    Any: all targets (quantized + float).
    """
    QConfigurable = auto()
    QNonConfigurable = auto()
    AnyQuantized = auto()
    Any = auto()


class Utilization(NamedTuple):
    """
    Utility container for a single resource utilization result.
    Supports sum, max, min over an iterable of Utilization objects.

    Args:
      size: parameters or activation tensor(s) size.
      bytes: memory utilization.
    """
    size: int
    bytes: Optional[float]

    def __add__(self, other: 'Utilization') -> 'Utilization':
        return Utilization(self.size + other.size, self.bytes + other.bytes)

    def __radd__(self, other: Union['Utilization', Literal[0]]):
        # Needed for sum (with default start_value=0).
        if other == 0:
            return self
        return self + other    # pragma: no cover

    def __gt__(self, other: 'Utilization'):
        # Needed for max. Compare by bytes.
        return self.bytes > other.bytes

    def __lt__(self, other: 'Utilization'):
        # Needed for min. Compare by bytes.
        return self.bytes < other.bytes    # pragma: no cover


class ResourceUtilizationCalculator:
    """ Resource utilization calculator. """

    _bitwidth_mode_fn = {
        BitwidthMode.QMaxBit: max,
        BitwidthMode.QMinBit: min,
    }

    def __init__(self, graph: Graph, fw_impl: FrameworkImplementation, fw_info: FrameworkInfo):
        self.graph = graph
        self.fw_impl = fw_impl
        self.fw_info = fw_info

        # Currently we go over the full graph even if utilization won't be requested for all nodes.
        # We could fill the cache on the fly only for requested nodes, but it's probably negligible.
        self._act_tensors_size = {}
        self._params_cnt = {}
        for n in graph.nodes:
            self._act_tensors_size[n] = n.get_total_output_params()
            self._params_cnt[n] = {k: v.size for k, v in n.weights.items()}
        self._cuts: Optional[Dict[Cut, List[BaseNode]]] = None

    @property
    def cuts(self) -> Dict[Cut, List[BaseNode]]:
        """ Compute if needed and return graph cuts and their memory element nodes. """
        if self._cuts is None:
            memory_graph = MemoryGraph(deepcopy(self.graph))
            _, _, cuts = compute_graph_max_cut(memory_graph)
            if cuts is None:    # pragma: no cover
                raise RuntimeError("Failed to calculate activation memory cuts for graph.")  # pragma: no cover
            cuts = [cut for cut in cuts if cut.mem_elements.elements]
            # cache cuts nodes for future use, so do not filter by target
            self._cuts = {cut: [self.graph.find_node_by_name(m.node_name)[0] for m in cut.mem_elements.elements]
                          for cut in cuts}
        return self._cuts

    def compute_resource_utilization(self,
                                     target_criterion: TargetInclusionCriterion,
                                     bitwidth_mode: BitwidthMode,
                                     act_qcs: Optional[Dict[BaseNode, NodeActivationQuantizationConfig]] = None,
                                     w_qcs: Optional[Dict[BaseNode, NodeWeightsQuantizationConfig]] = None,
                                     ru_targets: Iterable[RUTarget] = None) -> ResourceUtilization:
        """
        Compute network's resource utilization.

        Args:
            target_criterion: criterion to include targets for computation (applies to weights, activation).
            bitwidth_mode: bit-width mode for computation.
            act_qcs: custom activations quantization configuration. Should be provided for custom bit mode only.
              In custom mode, must provide configuration for all configurable activations. For non-configurable
              activations, if not provided, the default configuration will be extracted from the node.
            w_qcs: custom weights quantization configuration. Should be provided for custom bit mode only.
              In custom mode, must provide configuration for all configurable weights. For non-configurable
              weights, if not provided, the default configuration will be extracted from the node.
            ru_targets: metrics to include for computation. If None, all metrics are calculated.

        Returns:
            Resource utilization object.
        """
        ru_targets = set(ru_targets) if ru_targets else set(RUTarget)

        if w_qcs is not None and not self.is_custom_weights_config_applicable(ru_targets):
            raise ValueError('Weight configuration passed but no relevant metric requested.')
        if act_qcs is not None and not self.is_custom_activation_config_applicable(ru_targets):
            raise ValueError('Activation configuration passed but no relevant metric requested.')

        w_total, a_total = None, None
        if {RUTarget.WEIGHTS, RUTarget.TOTAL}.intersection(ru_targets):
            w_total, *_ = self.compute_weights_utilization(target_criterion, bitwidth_mode, w_qcs)

        if {RUTarget.ACTIVATION, RUTarget.TOTAL}.intersection(ru_targets):
            Logger.warning("Using an experimental feature max-cut for activation memory utilization estimation.")
            a_total = self.compute_activations_utilization(target_criterion, bitwidth_mode, act_qcs)

        ru = ResourceUtilization()
        if RUTarget.WEIGHTS in ru_targets:
            ru.weights_memory = w_total
        if RUTarget.ACTIVATION in ru_targets:
            ru.activation_memory = a_total
        if RUTarget.TOTAL in ru_targets:
            ru.total_memory = w_total + a_total
        if RUTarget.BOPS in ru_targets:
            ru.bops, _ = self.compute_bops(target_criterion=target_criterion,
                                           bitwidth_mode=bitwidth_mode, act_qcs=act_qcs, w_qcs=w_qcs)

        assert ru.get_restricted_targets() == set(ru_targets), 'Mismatch between the number of requested and computed metrics'
        return ru

    def compute_weights_utilization(self,
                                    target_criterion: TargetInclusionCriterion,
                                    bitwidth_mode: BitwidthMode,
                                    w_qcs: Optional[Dict[BaseNode, NodeWeightsQuantizationConfig]] = None) \
            -> Tuple[float, Dict[BaseNode, Utilization], Dict[BaseNode, Dict[str, Utilization]]]:
        """
        Compute graph's weights resource utilization.

        Args:
            target_criterion: criterion to include targets for computation.
            bitwidth_mode: bit-width mode for computation.
            w_qcs: custom weights quantization configuration. Should be provided for custom bit mode only.
              In custom mode, must provide configuration for all configurable weights. For non-configurable
              weights, if not provided, the default configuration will be extracted from the node.

        Returns:
            - Total weights utilization of the network.
            - Per node total weights utilization. Dict keys are nodes in a topological order.
            - Detailed per node per weight attribute utilization. Dict keys are nodes in a topological order.
        """
        nodes = self._get_target_weight_nodes(target_criterion, include_reused=False)
        if not nodes:
            return 0, {}, {}

        util_per_node: Dict[BaseNode, Utilization] = {}
        util_per_node_per_weight = {}

        for n in self._topo_sort(nodes):
            w_qc = w_qcs.get(n) if w_qcs else None
            node_weights_util, per_weight_util = self.compute_node_weights_utilization(n, target_criterion,
                                                                                       bitwidth_mode, w_qc)
            util_per_node[n] = node_weights_util
            util_per_node_per_weight[n] = per_weight_util

        total_util = sum(util_per_node.values())
        return total_util.bytes, util_per_node, util_per_node_per_weight

    def compute_node_weights_utilization(self,
                                         n: BaseNode,
                                         target_criterion: TargetInclusionCriterion,
                                         bitwidth_mode: BitwidthMode,
                                         qc: NodeWeightsQuantizationConfig)\
            -> Tuple[Utilization, Dict[str, Utilization]]:
        """
        Compute resource utilization for weights of a node.

        Args:
            n: node.
            target_criterion: criterion to include weights for computation.
            bitwidth_mode: bit-width mode for the computation.
            qc: custom weights quantization configuration. Should be provided for custom bit mode only.
              In custom mode, must provide configuration for all configurable weights. For non-configurable
              weights, if not provided, the default configuration will be extracted from the node.

        Returns:
            - Node's total weights utilization.
            - Detailed per weight attribute utilization.
        """
        weight_attrs = self._get_target_weight_attrs(n, target_criterion)
        if not weight_attrs:    # pragma: no cover
            return Utilization(0, 0), {}

        attr_util = {}
        for attr in weight_attrs:
            size = self._params_cnt[n][attr]
            nbits = self._get_weight_nbits(n, attr, bitwidth_mode, qc)
            bytes_ = size * nbits / 8
            attr_util[attr] = Utilization(size, bytes_)

        total_weights: Utilization = sum(attr_util.values())    # type: ignore
        return total_weights, attr_util

    def compute_activations_utilization(self,
                                        target_criterion: TargetInclusionCriterion,
                                        bitwidth_mode: BitwidthMode,
                                        act_qcs: Optional[Dict[BaseNode, NodeActivationQuantizationConfig]] = None):
        """
        Compute total activations utilization in the graph.

        Args:
            target_criterion: criterion to include weights for computation.
            bitwidth_mode: bit-width mode for the computation.
            act_qcs: custom activations quantization configuration. Should be provided for custom bit mode only.
              In custom mode, must provide configuration for all configurable activations. For non-configurable
              activations, if not provided, the default configuration will be extracted from the node.

        Returns:
            Total activation utilization of the network.
        """
        return self.compute_activation_utilization_by_cut(target_criterion, bitwidth_mode, act_qcs)[0]

    def compute_activation_utilization_by_cut(self,
                                              target_criterion: TargetInclusionCriterion,
                                              bitwidth_mode: BitwidthMode,
                                              act_qcs: Optional[Dict[BaseNode, NodeActivationQuantizationConfig]]) \
            -> Tuple[float, Dict[Cut, Utilization], Dict[Cut, Dict[BaseNode, Utilization]]]:
        """
        Compute graph activation cuts utilization.

        Args:
            target_criterion: criterion to include weights for computation.
            bitwidth_mode: bit-width mode for the computation.
            act_qcs: custom activations quantization configuration. Should be provided for custom bit mode only.
              In custom mode, must provide configuration for all configurable activations. For non-configurable
              activations, if not provided, the default configuration will be extracted from the node.

        Returns:
            - Total activation utilization of the network.
            - Total activation utilization per cut.
            - Detailed activation utilization per cut per node.
        """
        if target_criterion != TargetInclusionCriterion.AnyQuantized:    # pragma: no cover
            raise NotImplementedError('Computing MaxCut activation utilization is currently only supported for quantized targets.')

        graph_target_nodes = self._get_target_activation_nodes(target_criterion, include_reused=True)
        # if there are no target activations in the graph, don't waste time looking for cuts
        if not graph_target_nodes:
            return 0, {}, {}

        util_per_cut: Dict[Cut, Utilization] = {}    # type: ignore
        util_per_cut_per_node = defaultdict(dict)
        for cut in self.cuts:
            cut_target_nodes = self._get_cut_target_nodes(cut, target_criterion)
            if not cut_target_nodes:
                continue
            for n in cut_target_nodes:
                qc = act_qcs.get(n) if act_qcs else None
                util_per_cut_per_node[cut][n] = self.compute_node_activation_tensor_utilization(n, target_criterion,
                                                                                                bitwidth_mode, qc)
            util_per_cut[cut] = sum(util_per_cut_per_node[cut].values())    # type: ignore

        total_util = max(util_per_cut.values())
        return total_util.bytes, util_per_cut, util_per_cut_per_node

    def compute_activation_tensors_utilization(self,
                                               target_criterion: TargetInclusionCriterion,
                                               bitwidth_mode: BitwidthMode,
                                               act_qcs: Optional[Dict[BaseNode, NodeActivationQuantizationConfig]] = None,
                                               include_reused=False) \
            -> Tuple[float, Dict[BaseNode, Utilization]]:    # pragma: no cover
        """
        Compute resource utilization for graph's activations tensors.

        Args:
            target_criterion: criterion to include weights for computation.
            bitwidth_mode: bit-width mode for the computation.
            act_qcs: custom activations quantization configuration. Should be provided for custom bit mode only.
              In custom mode, must provide configuration for all configurable activations. For non-configurable
              activations, if not provided, the default configuration will be extracted from the node.
            include_reused: whether to include reused nodes.
        Returns:
            - Total activation utilization of the network.
            - Detailed utilization per node. Dict keys are nodes in a topological order.

        """
        nodes = self._get_target_activation_nodes(target_criterion, include_reused=include_reused)
        if not nodes:
            return 0, {}

        util_per_node: Dict[BaseNode, Utilization] = {}
        for n in self._topo_sort(nodes):
            qc = act_qcs.get(n) if act_qcs else None
            util = self.compute_node_activation_tensor_utilization(n, None, bitwidth_mode, qc)
            util_per_node[n] = util

        total_util = max(util_per_node.values())
        return total_util.bytes, util_per_node

    def compute_node_activation_tensor_utilization(self,
                                                   n: BaseNode,
                                                   target_criterion: Optional[TargetInclusionCriterion],
                                                   bitwidth_mode: BitwidthMode,
                                                   qc: Optional[NodeActivationQuantizationConfig]) -> Utilization:
        """
        Compute activation resource utilization for a node.

        Args:
            n: node.
            target_criterion: criterion to include nodes for computation. If None, will skip the check.
            bitwidth_mode: bit-width mode for the computation.
            qc: activation quantization config for the node. Should be provided only in custom bit mode.
              In custom mode, must be provided if the activation is configurable. For non-configurable activation, if
              not passed, the default configuration will be extracted from the node.
        Returns:
            Node's activation utilization.
        """
        if target_criterion:
            nodes = self._get_target_activation_nodes(target_criterion=target_criterion, include_reused=True, nodes=[n])
            if not nodes:    # pragma: no cover
                return Utilization(0, 0)

        size = self._act_tensors_size[n]
        nbits = self._get_activation_nbits(n, bitwidth_mode, qc)
        bytes_ = size * nbits / 8
        return Utilization(size, bytes_)

    def compute_bops(self,
                     target_criterion: TargetInclusionCriterion,
                     bitwidth_mode: BitwidthMode,
                     act_qcs: Optional[Dict[BaseNode, NodeActivationQuantizationConfig]] = None,
                     w_qcs: Optional[Dict[BaseNode, NodeWeightsQuantizationConfig]] = None) \
            -> Tuple[int, Dict[BaseNode, int]]:
        """
        Compute bit operations based on nodes with kernel.
        Note that 'target_criterion' applies to weights, and BOPS are computed for the selected nodes regardless
        of the input activation quantization or lack thereof.

        Args:
            target_criterion: criterion to include nodes for computation.
            bitwidth_mode: bit-width mode for computation.
            act_qcs: custom activations quantization configuration. Should be provided for custom bit mode only.
              In custom mode, must provide configuration for all configurable activations. For non-configurable
              activations, if not provided, the default configuration will be extracted from the node.
            w_qcs: custom weights quantization configuration. Should be provided for custom bit mode only.
              In custom mode, must provide configuration for all configurable weights. For non-configurable
              weights, if not provided, the default configuration will be extracted from the node.

        Returns:
            - Total BOPS count of the network.
            - Detailed BOPS count per node.
        """
        if target_criterion != TargetInclusionCriterion.AnyQuantized:    # pragma: no cover
            raise NotImplementedError('BOPS computation is currently only supported for quantized targets.')

        nodes = self._get_target_weight_nodes(target_criterion, include_reused=True)
        # filter out nodes with only positional weights # TODO add as arg to get target nodes
        nodes = [n for n in nodes if n.has_kernel_weight_to_quantize(self.fw_info)]

        nodes_bops = {}
        for n in nodes:
            w_qc = w_qcs.get(n) if w_qcs else None
            nodes_bops[n] = self.compute_node_bops(n, bitwidth_mode, act_qcs=act_qcs, w_qc=w_qc)

        return sum(nodes_bops.values()), nodes_bops

    def compute_node_bops(self,
                          n: BaseNode,
                          bitwidth_mode: BitwidthMode,
                          act_qcs: Optional[Dict[BaseNode, NodeActivationQuantizationConfig]] = None,
                          w_qc: Optional[NodeWeightsQuantizationConfig] = None) -> Union[float, int]:
        """
        Compute Bit Operations of a node.

        Args:
            n: node.
            bitwidth_mode: bit-width mode for the computation.
            act_qcs: custom activations quantization configuration. Should be provided for custom bit mode only.
              In custom mode, must provide configuration for all configurable activations. For non-configurable
              activations, if not provided, the default configuration will be extracted from the node.
            w_qc: weights quantization config for the node. Should be provided only in custom bit mode.
              Must provide configuration for all configurable weights. For non-configurable weights, will use the
              provided configuration if found, or extract the default configuration from the node otherwise.

        Returns:
            Node's BOPS count.
        """
        node_mac = self.fw_impl.get_node_mac_operations(n, self.fw_info)
        if node_mac == 0:    # pragma: no cover
            return node_mac

        incoming_edges = self.graph.incoming_edges(n, sort_by_attr=EDGE_SINK_INDEX)
        # TODO temporary adding this for const_representation test in torch which has Linear with const input
        if not incoming_edges:
            return 0
        assert len(incoming_edges) == 1, \
            f'Unexpected number of inputs {len(incoming_edges)} for BOPS calculation. Expected 1.'
        input_act_node = incoming_edges[0].source_node
        act_qc = act_qcs.get(input_act_node) if act_qcs else None
        a_nbits = self._get_activation_nbits(input_act_node, bitwidth_mode, act_qc)

        kernel_attrs = self.fw_info.get_kernel_op_attributes(n.type)
        if len(kernel_attrs) > 1:    # pragma: no cover
            raise NotImplementedError('Multiple kernel attributes are not supported for BOPS computation.')
        kernel_attr = kernel_attrs[0]
        w_nbits = self._get_weight_nbits(n, kernel_attr, bitwidth_mode, w_qc)

        node_bops = a_nbits * w_nbits * node_mac
        return node_bops

    def is_custom_weights_config_applicable(self, ru_targets: Set[RUTarget]) -> bool:
        """ Whether custom configuration for weights is compatible with the requested targets."""
        return bool({RUTarget.WEIGHTS, RUTarget.TOTAL, RUTarget.BOPS}.intersection(ru_targets))

    def is_custom_activation_config_applicable(self, ru_targets: Set[RUTarget]) -> bool:
        """ Whether custom configuration for activations is compatible with the requested targets."""
        return bool({RUTarget.ACTIVATION, RUTarget.TOTAL, RUTarget.BOPS}.intersection(ru_targets))

    def _get_cut_target_nodes(self, cut: Cut, target_criterion: TargetInclusionCriterion) -> List[BaseNode]:
        """
        Retrieve target nodes from a cut filtered by a criterion.

        Args:
            cut: a graph cut.
            target_criterion: criterion to include nodes for computation.

        Returns:
            A list of target nodes from a cut.
        """
        cut_nodes = self.cuts[cut]
        return self._get_target_activation_nodes(target_criterion, include_reused=True, nodes=cut_nodes)

    def _get_target_weight_nodes(self,
                                 target_criterion: TargetInclusionCriterion,
                                 include_reused: bool) -> List[BaseNode]:
        """
        Collect nodes to include in weights utilization computation.

        Args:
            target_criterion: criterion to include weights for computation.
            include_reused: whether to include reused nodes.

        Returns:
            Target nodes.
        """
        if target_criterion == TargetInclusionCriterion.QConfigurable:
            nodes = self.graph.get_weights_configurable_nodes(self.fw_info, include_reused_nodes=include_reused)
        elif target_criterion == TargetInclusionCriterion.AnyQuantized:
            nodes = [n for n in self.graph if n.has_any_weight_attr_to_quantize()]
        elif target_criterion == TargetInclusionCriterion.QNonConfigurable:
            # TODO this is wrong. Need to look at specific weights and not the whole node (if w1 is configurable and w2
            #  is non-configurable we want to discover the node both as configurable and non-configurable)
            quantized = [n for n in self.graph if n.has_any_weight_attr_to_quantize()]
            configurable = self.graph.get_weights_configurable_nodes(self.fw_info, include_reused_nodes=include_reused)
            nodes = [n for n in quantized if n not in configurable]
        elif target_criterion == TargetInclusionCriterion.Any:    # pragma: no cover
            nodes = list(self.graph.nodes)
        else:    # pragma: no cover
            raise ValueError(f'Unknown {target_criterion}.')

        if not include_reused:
            nodes = [n for n in nodes if not n.reuse]
        return nodes

    def _get_target_weight_attrs(self, n: BaseNode, target_criterion: TargetInclusionCriterion) -> List[str]:
        """
        Collect weight attributes of a node per criterion.

        Args:
            n: node.
            target_criterion: selection criterion.

        Returns:
            Selected weight attributes names.
        """
        weight_attrs = n.get_node_weights_attributes()
        if target_criterion == TargetInclusionCriterion.QConfigurable:
            weight_attrs = [attr for attr in weight_attrs if n.is_configurable_weight(attr)]
        elif target_criterion == TargetInclusionCriterion.AnyQuantized:
            weight_attrs = [attr for attr in weight_attrs if n.is_weights_quantization_enabled(attr)]
        elif target_criterion == TargetInclusionCriterion.QNonConfigurable:
            quantized = [attr for attr in weight_attrs if n.is_weights_quantization_enabled(attr)]
            configurable = [attr for attr in weight_attrs if n.is_configurable_weight(attr)]
            weight_attrs = [attr for attr in quantized if attr not in configurable]
        elif target_criterion != TargetInclusionCriterion.Any:    # pragma: no cover
            raise ValueError(f'Unknown {target_criterion}')
        return weight_attrs

    def _topo_sort(self, nodes: Sequence[BaseNode]) -> List[BaseNode]:
        """
        Sort nodes in a topological order (based on graph's nodes).

        Args:
            nodes: nodes to sort.

        Returns:
            Nodes in topological order.
        """
        graph_topo_nodes = self.graph.get_topo_sorted_nodes()
        topo_nodes = [n for n in graph_topo_nodes if n in nodes]
        if len(topo_nodes) != len(nodes):    # pragma: no cover
            missing_nodes = [n for n in nodes if n not in topo_nodes]
            raise ValueError(f'Could not topo-sort, nodes {missing_nodes} do not match the graph nodes.')
        return topo_nodes

    def _get_target_activation_nodes(self,
                                     target_criterion: TargetInclusionCriterion,
                                     include_reused: bool,
                                     nodes: Optional[List[BaseNode]] = None) -> List[BaseNode]:
        """
        Collect nodes to include in activation utilization computation.

        Args:
            target_criterion: criterion to include activations for computation.
            include_reused: whether to include reused nodes.
            nodes: nodes to filter target nodes from. By default, uses the graph nodes.

        Returns:
            Selected nodes.
        """
        nodes = nodes or self.graph.nodes
        if target_criterion == TargetInclusionCriterion.QConfigurable:    # pragma: no cover
            nodes = [n for n in nodes if n.has_configurable_activation()]
        elif target_criterion == TargetInclusionCriterion.AnyQuantized:
            nodes = [n for n in nodes if n.is_activation_quantization_enabled()]
        elif target_criterion == TargetInclusionCriterion.QNonConfigurable:    # pragma: no cover
            nodes = [n for n in nodes if n.is_activation_quantization_enabled() and not n.has_configurable_activation()]
        elif target_criterion != TargetInclusionCriterion.Any:    # pragma: no cover
            raise ValueError(f'Unknown {target_criterion}.')
        if not include_reused:    # pragma: no cover
            nodes = [n for n in nodes if not n.reuse]
        return nodes

    @classmethod
    def _get_activation_nbits(cls,
                              n: BaseNode,
                              bitwidth_mode: BitwidthMode,
                              act_qc: Optional[NodeActivationQuantizationConfig]) -> int:
        """
        Get activation bit-width for a node according to the requested bit-width mode.

        Args:
            n: node.
            bitwidth_mode: bit-width mode for computation.
            act_qc: activation quantization config for the node. Should be provided only in custom bit mode.
              In custom mode, must be provided if the activation is configurable. For non-configurable activation, if
              not passed, the default configuration will be extracted from the node.

        Returns:
            Activation bit-width.
        """
        if act_qc:
            if bitwidth_mode != BitwidthMode.QCustom:    # pragma: no cover
                raise ValueError(f'Activation config is not expected for non-custom bit mode {bitwidth_mode}')
            return act_qc.activation_n_bits if act_qc.enable_activation_quantization else FLOAT_BITWIDTH

        if bitwidth_mode == BitwidthMode.Float or not n.is_activation_quantization_enabled():
            return FLOAT_BITWIDTH

        if bitwidth_mode == BitwidthMode.Q8Bit:
            return 8

        if bitwidth_mode in cls._bitwidth_mode_fn:
            candidates_nbits = [c.activation_quantization_cfg.activation_n_bits for c in n.candidates_quantization_cfg]
            return cls._bitwidth_mode_fn[bitwidth_mode](candidates_nbits)

        if bitwidth_mode in [BitwidthMode.QCustom, BitwidthMode.QDefaultSP]:
            qcs = n.get_unique_activation_candidates()
            if len(qcs) != 1:    # pragma: no cover
                raise ValueError(f'Could not retrieve the activation quantization candidate for node {n.name} '
                                 f'as it has {len(qcs)}!=1 unique candidates .')
            return qcs[0].activation_quantization_cfg.activation_n_bits

        raise ValueError(f'Unknown mode {bitwidth_mode}')    # pragma: no cover

    @classmethod
    def _get_weight_nbits(cls,
                          n: BaseNode,
                          w_attr: str,
                          bitwidth_mode: BitwidthMode,
                          w_qc: Optional[NodeWeightsQuantizationConfig]) -> int:
        """
        Get the bit-width of a specific weight of a node according to the requested bit-width mode.

        Args:
            n: node.
            w_attr: weight attribute.
            bitwidth_mode: bit-width mode for the computation.
            w_qc: weights quantization config for the node. Should be provided only in custom bit mode.
              Must provide configuration for all configurable weights. For non-configurable weights, will use the
              provided configuration if found, or extract the default configuration from the node otherwise.

        Returns:
            Weight bit-width.
        """
        if w_qc and w_qc.has_attribute_config(w_attr):
            if bitwidth_mode != BitwidthMode.QCustom:    # pragma: no cover
                raise ValueError('Weight config is not expected for non-custom bit mode {bitwidth_mode}')
            attr_cfg = w_qc.get_attr_config(w_attr)
            return attr_cfg.weights_n_bits if attr_cfg.enable_weights_quantization else FLOAT_BITWIDTH

        if bitwidth_mode == BitwidthMode.Float or not n.is_weights_quantization_enabled(w_attr):
            return FLOAT_BITWIDTH

        if bitwidth_mode == BitwidthMode.Q8Bit:
            return 8

        node_qcs = n.get_unique_weights_candidates(w_attr)
        w_qcs = [qc.weights_quantization_cfg.get_attr_config(w_attr) for qc in node_qcs]
        if bitwidth_mode in cls._bitwidth_mode_fn:
            return cls._bitwidth_mode_fn[bitwidth_mode]([qc.weights_n_bits for qc in w_qcs])

        if bitwidth_mode in [BitwidthMode.QCustom, BitwidthMode.QDefaultSP]:
            # if configuration was not passed and the weight has only one candidate, use it
            if len(w_qcs) != 1:    # pragma: no cover
                raise ValueError(f'Could not retrieve the quantization candidate for attr {w_attr} of node {n.name} '
                                 f'as it {len(w_qcs)}!=1 unique candidates.')
            return w_qcs[0].weights_n_bits

        raise ValueError(f'Unknown mode {bitwidth_mode.name}')    # pragma: no cover

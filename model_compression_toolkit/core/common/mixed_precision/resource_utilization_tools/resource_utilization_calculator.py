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
from collections import defaultdict
from copy import deepcopy
from enum import Enum, auto
from functools import lru_cache
from typing import Dict, NamedTuple, Optional, Tuple, List, Iterable, Union, Literal

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

    Size: tensors sizes.
    Float: float.
    MpMax: maximal bit-width mixed-precision configuration.
    MpMin: minimal bit-width mixed-precision configuration.
    MpCustom: explicitly provided bit-width configuration.
    SpDefault: single-precision configuration (for non-configurable quantization).
    """
    Size = auto()
    Float = auto()
    MpMax = auto()
    MpMin = auto()
    MpCustom = auto()
    SpDefault = auto()


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

    def by_bit_mode(self, bitwidth_mode: BitwidthMode) -> Union[int, float]:
        """ Retrieve value corresponding to the bit-width mode. """
        if bitwidth_mode == BitwidthMode.Size:
            return self.size
        return self.bytes

    @staticmethod
    def zero_utilization(bitwidth_mode: BitwidthMode) -> 'Utilization':
        """ Construct zero utilization object. """
        return Utilization(0, bytes=None if bitwidth_mode == BitwidthMode.Size else 0)

    def __add__(self, other: 'Utilization') -> 'Utilization':
        self._validate_pair(self, other)
        bytes_ = None if self.bytes is None else (self.bytes + other.bytes)
        return Utilization(self.size + other.size, bytes_)

    def __radd__(self, other: Union['Utilization', Literal[0]]):
        # Needed for sum (with default start_value=0).
        if other == 0:
            return self
        return self + other

    def __gt__(self, other: 'Utilization'):
        # Needed for max. Compare by bytes, if not defined then by size.
        self._validate_pair(self, other)
        if self.bytes is not None:
            return self.bytes > other.bytes
        return self.size > other.size

    def __lt__(self, other: 'Utilization'):
        self._validate_pair(self, other)
        # Needed for min. Compare by bytes, if not defined then by size.
        if self.bytes is not None:
            return self.bytes < other.bytes
        return self.size < other.size

    @staticmethod
    def _validate_pair(u1, u2):
        if [u1.bytes, u2.bytes].count(None) == 1:
            raise ValueError('bytes field must be set either by both or by none of the objects.')


class AggregationMethod(Enum):
    SUM = sum
    MAX = lambda seq: max(seq) if (seq := list(seq)) else 0    # walrus op for empty generator

    def __call__(self, *args, **kwarg):
        return self.value(*args, **kwarg)


# default aggregation methods
# TODO This is used by mp to use the same aggregation. Except that for total it must do its own thing (add indicators
#  to weights before summation). So maybe just get rid of it altogether? If it ever becomes configurable we can add it.
ru_target_aggregation_fn = {
    RUTarget.WEIGHTS: AggregationMethod.SUM,
    RUTarget.ACTIVATION: AggregationMethod.MAX,
    RUTarget.TOTAL: AggregationMethod.SUM,
    RUTarget.BOPS: AggregationMethod.SUM
}


_bitwidth_mode_fn = {
    BitwidthMode.MpMax: max,
    BitwidthMode.MpMin: min
}


class ResourceUtilizationCalculator:
    """ Resource utilization calculator. """

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
        self._cuts = None

    def compute_resource_utilization(self,
                                     target_criterion: TargetInclusionCriterion,
                                     bitwidth_mode: BitwidthMode,
                                     act_qcs: Optional[Dict[BaseNode, NodeActivationQuantizationConfig]] = None,
                                     w_qcs: Optional[Dict[BaseNode, NodeWeightsQuantizationConfig]] = None,
                                     metrics: Iterable[RUTarget] = None) -> ResourceUtilization:
        """
        Compute total resource utilization.

        Args:
            target_criterion: criterion to include targets for computation (applies to weights, activation).
            bitwidth_mode: bit-width mode for computation.
            act_qcs: activation quantization candidates for custom bit-width mode. Must provide configuration for all
              configurable activations.
            w_qcs: weights quantization candidates for custom bit-width mode. Must provide configuration for all
              configurable weights.
            metrics: metrics to include for computation. If None, all metrics are calculated.

        Returns:
            Resource utilization object.
        """
        metrics = set(metrics) if metrics else set(RUTarget)

        w_total, a_total = None, None
        if {RUTarget.WEIGHTS, RUTarget.TOTAL}.intersection(metrics):
            w_total, *_ = self.compute_weights_utilization(target_criterion, bitwidth_mode, w_qcs)
        elif w_qcs is not None:    # pragma: no cover
            raise ValueError('Weight configuration passed but no relevant metric requested.')

        if act_qcs and not {RUTarget.ACTIVATION, RUTarget.TOTAL}.intersection(metrics):    # pragma: no cover
            raise ValueError('Activation configuration passed but no relevant metric requested.')
        if RUTarget.ACTIVATION in metrics:
            a_total, *_ = self.compute_activations_utilization(target_criterion, bitwidth_mode, act_qcs)

        ru = ResourceUtilization()
        if RUTarget.WEIGHTS in metrics:
            ru.weights_memory = w_total
        if RUTarget.ACTIVATION in metrics:
            ru.activation_memory = a_total
        if RUTarget.TOTAL in metrics:
            # TODO use maxcut
            act_tensors_total, *_ = self.compute_activation_tensors_utilization(target_criterion, bitwidth_mode, act_qcs)
            ru.total_memory = w_total + act_tensors_total
        if RUTarget.BOPS in metrics:
            ru.bops, _ = self.compute_bops(target_criterion=target_criterion,
                                           bitwidth_mode=bitwidth_mode, act_qcs=act_qcs, w_qcs=w_qcs)

        assert ru.get_restricted_metrics() == set(metrics), 'Mismatch between the number of requested and computed metrics'
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
            w_qcs: weights quantization config per node for the custom bit mode. Must provide configuration for all
              configurable weights.

        Returns:
            - Total weights utilization.
            - Per node total utilization. Dict keys are nodes in a topological order.
            - Detailed per node per weight utilization. Dict keys are nodes in a topological order.
        """
        nodes = self._get_target_weight_nodes(target_criterion, include_reused=False)

        util_per_node: Dict[BaseNode, Utilization] = {}
        util_per_node_per_weight = {}

        for n in self._topo_sort(nodes):
            w_qc = w_qcs.get(n) if w_qcs else None
            node_weights_util, per_weight_util = self.compute_node_weights_utilization(n, target_criterion,
                                                                                       bitwidth_mode, w_qc)
            util_per_node[n] = node_weights_util
            util_per_node_per_weight[n] = per_weight_util

        aggregate_fn = ru_target_aggregation_fn[RUTarget.WEIGHTS]
        total_util = aggregate_fn(u.by_bit_mode(bitwidth_mode) for u in util_per_node.values())
        return total_util, util_per_node, util_per_node_per_weight

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
            qc: weight quantization config for the custom bit mode computation. Must provide configuration for all
              configurable weights.

        Returns:
            - Total utilization.
            - Detailed per weight utilization.
        """
        weight_attrs = self._get_target_weight_attrs(n, target_criterion)
        if not weight_attrs:    # pragma: no cover
            return Utilization.zero_utilization(bitwidth_mode, ), {}

        attr_util = {}
        for attr in weight_attrs:
            size = self._params_cnt[n][attr]
            bytes_ = None
            if bitwidth_mode != BitwidthMode.Size:
                nbits = self._get_weight_nbits(n, attr, bitwidth_mode, qc)
                bytes_ = size * nbits / 8
            attr_util[attr] = Utilization(size, bytes_)

        total_weights = sum(attr_util.values())
        return total_weights, attr_util

    def compute_activations_utilization(self,
                                        target_criterion: TargetInclusionCriterion,
                                        bitwidth_mode: BitwidthMode,
                                        act_qcs: Optional[Dict[BaseNode, NodeActivationQuantizationConfig]] = None):
        return self.compute_cut_activation_utilization(target_criterion, bitwidth_mode, act_qcs)

    def compute_cut_activation_utilization(self,
                                           target_criterion: TargetInclusionCriterion,
                                           bitwidth_mode: BitwidthMode,
                                           act_qcs: Optional[Dict[BaseNode, NodeActivationQuantizationConfig]]) \
            -> Tuple[float, Dict[Cut, Utilization], Dict[Cut, Dict[BaseNode, Utilization]]]:
        """
        Calculate graph activation cuts utilization.

        Args:
            target_criterion: criterion to include weights for computation.
            bitwidth_mode: bit-width mode for the computation.
            act_qcs: custom configuration for the custom bit mode. Must provide configuration for all configurable
              activations.

        Returns:
            - Total utilization.
            - Total utilization per cut.
            - Detailed utilization per cut per node.
        """
        if target_criterion != TargetInclusionCriterion.AnyQuantized:    # pragma: no cover
            raise NotImplementedError('Computing MaxCut activation utilization is currently only supported for quantized targets.')

        graph_target_nodes = self._get_target_activation_nodes(target_criterion, include_reused=True)
        # if there are no target activations in the graph, don't waste time looking for cuts
        if not graph_target_nodes:
            return 0, {}, {}

        if self._cuts is None:
            memory_graph = MemoryGraph(deepcopy(self.graph))
            _, _, cuts = compute_graph_max_cut(memory_graph)
            if cuts is None:
                raise RuntimeError("Failed to calculate activation memory cuts for graph.")  # pragma: no cover
            cuts = [cut for cut in cuts if cut.mem_elements.elements]
            # cache cuts nodes for future use, so do not filter by target
            self._cuts = {cut: [self.graph.find_node_by_name(m.node_name)[0] for m in cut.mem_elements.elements]
                          for cut in cuts}

        util_per_cut: Dict[Cut, Utilization] = {}    # type: ignore
        util_per_cut_per_node = defaultdict(dict)
        for cut in self._cuts:
            cut_target_nodes = [n for n in self._cuts[cut] if n in graph_target_nodes]
            if not cut_target_nodes:
                continue
            for n in cut_target_nodes:
                qc = act_qcs.get(n) if act_qcs else None
                util_per_cut_per_node[cut][n] = self.compute_node_activation_tensor_utilization(n, target_criterion,
                                                                                                bitwidth_mode, qc)
            util_per_cut[cut] = sum(util_per_cut_per_node[cut].values())    # type: ignore

        aggregate_fn = ru_target_aggregation_fn[RUTarget.ACTIVATION]
        total_util = aggregate_fn(u.by_bit_mode(bitwidth_mode) for u in util_per_cut.values())
        return total_util, util_per_cut, util_per_cut_per_node

    def compute_activation_tensors_utilization(self,
                                               target_criterion: TargetInclusionCriterion,
                                               bitwidth_mode: BitwidthMode,
                                               act_qcs: Optional[Dict[BaseNode, NodeActivationQuantizationConfig]] = None,
                                               include_reused=False) \
            -> Tuple[float, Dict[BaseNode, Utilization]]:
        """
        Compute resource utilization for graph's activations tensors.

        Args:
            target_criterion: criterion to include weights for computation.
            bitwidth_mode: bit-width mode for the computation.
            act_qcs: custom configuration for the custom bit mode. Must provide configuration for all configurable
              activations.
            include_reused: whether to include reused nodes.
        Returns:
            - Total activation utilization.
            - Detailed utilization per node. Dict keys are nodes in a topological order.

        """
        nodes = self._get_target_activation_nodes(target_criterion, include_reused=include_reused)
        util_per_node: Dict[BaseNode, Utilization] = {}
        for n in self._topo_sort(nodes):
            qc = act_qcs.get(n) if act_qcs else None
            util = self.compute_node_activation_tensor_utilization(n, None, bitwidth_mode, qc)
            util_per_node[n] = util

        aggregate_fn = ru_target_aggregation_fn[RUTarget.ACTIVATION]
        total_util = aggregate_fn(u.by_bit_mode(bitwidth_mode) for u in util_per_node.values())
        return total_util, util_per_node

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
            qc: activation quantization config for the custom bit mode. Must be provided for a configurable activation.

        Returns:
            Node's activation utilization.
        """
        if target_criterion:
            nodes = self._get_target_activation_nodes(target_criterion=target_criterion, include_reused=True, nodes=[n])
            if not nodes:    # pragma: no cover
                return Utilization.zero_utilization(bitwidth_mode)

        size = self._act_tensors_size[n]
        bytes_ = None
        if bitwidth_mode != BitwidthMode.Size:
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

        Args:
            target_criterion: criterion to include nodes for computation.
            bitwidth_mode: bit-width mode for computation.
            act_qcs: activation quantization candidates for custom bit-width mode. Must provide configuration for all
              configurable activations.
            w_qcs: weights quantization candidates for custom bit-width mode. Must provide configuration for all
              configurable weights.

        Returns:
            - Total BOPS count.
            - Detailed BOPS count per node.
        """
        # currently we compute bops for all nodes with quantized weights, regardless of whether the input
        # activation is quantized.
        if target_criterion != TargetInclusionCriterion.AnyQuantized:    # pragma: no cover
            raise NotImplementedError('BOPS computation is currently only supported for quantized targets.')

        nodes = [n for n in self.graph.nodes if n.has_kernel_weight_to_quantize(self.fw_info)]
        nodes_bops = {}
        for n in nodes:
            w_qc = w_qcs.get(n) if w_qcs else None
            nodes_bops[n] = self.compute_node_bops(n, bitwidth_mode, act_qcs=act_qcs, w_qc=w_qc)

        aggregate_fn = ru_target_aggregation_fn[RUTarget.BOPS]
        return aggregate_fn(nodes_bops.values()), nodes_bops

    def compute_node_bops(self,
                          n: BaseNode,
                          bitwidth_mode: BitwidthMode,
                          act_qcs: Optional[Dict[BaseNode, NodeActivationQuantizationConfig]] = None,
                          w_qc: Optional[NodeWeightsQuantizationConfig] = None) -> int:
        """
        Compute Bit Operations of a node.

        Args:
            n: node.
            bitwidth_mode: bit-width mode for the computation.
            act_qcs: nodes activation quantization configuration for the custom bit mode. Must provide configuration for all
              configurable activations.
            w_qc: weights quantization config for the node for the custom bit mode. Must provide configuration for all
              configurable weights.

        Returns:
            BOPS count.
        """
        node_mac = self.fw_impl.get_node_mac_operations(n, self.fw_info)
        if node_mac == 0 or bitwidth_mode == BitwidthMode.Size:    # pragma: no cover
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
        if len(kernel_attrs) > 1:
            raise NotImplementedError('Multiple kernel attributes are not supported for BOPS computation.')
        kernel_attr = kernel_attrs[0]
        w_nbits = self._get_weight_nbits(n, kernel_attr, bitwidth_mode, w_qc)

        node_bops = a_nbits * w_nbits * node_mac
        return node_bops

    @lru_cache
    def _get_cut_target_nodes(self, cut: Cut, target_criterion: TargetInclusionCriterion) -> List[BaseNode]:
        """
        Retrieve target nodes from a cut filtered by a criterion.

        Args:
            cut: a graph cut.
            target_criterion: criterion to include nodes for computation.

        Returns:
            A list of target nodes from a cut.
        """
        cut_nodes = [self.graph.find_node_by_name(e.node_name)[0] for e in cut.mem_elements.elements]
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
            # TODO this is wrong. Need to look at specific weights and not the whole node
            quantized = [n for n in self.graph if n.has_any_weight_attr_to_quantize()]
            configurable = self.graph.get_weights_configurable_nodes(self.fw_info, include_reused_nodes=include_reused)
            nodes = [n for n in quantized if n not in configurable]
        elif target_criterion == TargetInclusionCriterion.Any:
            nodes = self.graph.nodes
        else:
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
        elif target_criterion != TargetInclusionCriterion.Any:
            raise ValueError(f'Unknown {target_criterion}')
        return weight_attrs

    def _topo_sort(self, nodes):
        """ Sort nodes in a topological order (based on graph's nodes). """
        graph_topo_nodes = self.graph.get_topo_sorted_nodes()
        topo_nodes = [n for n in graph_topo_nodes if n in nodes]
        if len(topo_nodes) != len(nodes):
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
        if target_criterion == TargetInclusionCriterion.QConfigurable:
            nodes = [n for n in nodes if n.has_configurable_activation()]
        elif target_criterion == TargetInclusionCriterion.AnyQuantized:
            nodes = [n for n in nodes if n.is_activation_quantization_enabled()]
        elif target_criterion == TargetInclusionCriterion.QNonConfigurable:
            nodes = [n for n in nodes if n.is_activation_quantization_enabled() and not n.has_configurable_activation()]
        elif target_criterion != TargetInclusionCriterion.Any:
            raise ValueError(f'Unknown {target_criterion}.')
        if not include_reused:
            nodes = [n for n in nodes if not n.reuse]
        return nodes

    @staticmethod
    def _get_activation_nbits(n: BaseNode,
                              bitwidth_mode: BitwidthMode,
                              act_qc: Optional[NodeActivationQuantizationConfig]) -> int:
        """
        Get activation bit-width for a node according to the requested bit-width mode.

        Args:
            n: node.
            bitwidth_mode: bit-width mode for computation.
            act_qc: quantization candidate for the custom bit mode. Must be provided for a configurable activation.

        Returns:
            Activation bit-width.
        """
        if bitwidth_mode == BitwidthMode.Size:
            raise ValueError(f'nbits is not defined for {bitwidth_mode}.')

        if act_qc:
            if bitwidth_mode != BitwidthMode.MpCustom or not n.is_activation_quantization_enabled():
                raise ValueError(
                    f'Activation config is not expected for non-custom bit mode or for un-quantized activation.'
                    f'Mode: {bitwidth_mode}, quantized activation: {n.is_activation_quantization_enabled()}'
                )
            assert act_qc.enable_activation_quantization
            return act_qc.activation_n_bits

        if bitwidth_mode == BitwidthMode.Float or not n.is_activation_quantization_enabled():
            return FLOAT_BITWIDTH

        if bitwidth_mode in _bitwidth_mode_fn:
            candidates_nbits = [c.activation_quantization_cfg.activation_n_bits for c in n.candidates_quantization_cfg]
            return _bitwidth_mode_fn[bitwidth_mode](candidates_nbits)

        if bitwidth_mode in [BitwidthMode.MpCustom, BitwidthMode.SpDefault]:
            qcs = n.get_unique_activation_candidates()
            if len(qcs) != 1:
                raise ValueError(f'Could not retrieve the activation quantization candidate for node {n.name} '
                                 f'as it has {len(qcs)}!=1 unique candidates .')
            return qcs[0].activation_quantization_cfg.activation_n_bits

        raise ValueError(f'Unknown mode {bitwidth_mode}')

    @staticmethod
    def _get_weight_nbits(n, w_attr: str, bitwidth_mode: BitwidthMode, w_qc: Optional[NodeWeightsQuantizationConfig]) -> int:
        """
        Get the bit-width of a specific weight of a node according to the requested bit-width mode.

        Args:
            n: node.
            w_attr: weight attribute.
            bitwidth_mode: bit-width mode for the computation.
            w_qc: weights quantization config for the node for the custom bit mode. Must provide configuration for all
              configurable weights.

        Returns:
            Weight bit-width.
        """
        if bitwidth_mode == BitwidthMode.Size:
            raise ValueError(f'nbits is not defined for {bitwidth_mode}.')

        if w_qc and w_qc.has_attribute_config(w_attr):
            if bitwidth_mode != BitwidthMode.MpCustom or not n.is_weights_quantization_enabled(w_attr):
                raise ValueError('Weight config is not expected for non-custom bit mode or for un-quantized weight.'
                                 f'Bit mode: {bitwidth_mode}, quantized attr {w_attr}: '
                                 f'{n.is_weights_quantization_enabled(w_attr)}')
            attr_cfg = w_qc.get_attr_config(w_attr)
            assert attr_cfg.enable_weights_quantization
            return attr_cfg.weights_n_bits

        if bitwidth_mode == BitwidthMode.Float or not n.is_weights_quantization_enabled(w_attr):
            return FLOAT_BITWIDTH

        node_qcs = n.get_unique_weights_candidates(w_attr)
        w_qcs = [qc.weights_quantization_cfg.get_attr_config(w_attr) for qc in node_qcs]
        if bitwidth_mode in _bitwidth_mode_fn:
            return _bitwidth_mode_fn[bitwidth_mode]([qc.weights_n_bits for qc in w_qcs])

        if bitwidth_mode in [BitwidthMode.MpCustom, BitwidthMode.SpDefault]:
            # if configuration was not passed and the weight has only one candidate, use it
            if len(w_qcs) != 1:
                raise ValueError(f'Could not retrieve the quantization candidate for attr {w_attr} of node {n.name} '
                                 f'as it {len(w_qcs)}!=1 unique candidates.')
            return w_qcs[0].weights_n_bits

        raise ValueError(f'Unknown mode {bitwidth_mode.name}')

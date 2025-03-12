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
from typing import Dict, NamedTuple, Optional, Tuple, List, Iterable, Union, Literal, Sequence

from model_compression_toolkit.constants import FLOAT_BITWIDTH
from model_compression_toolkit.core import FrameworkInfo
from model_compression_toolkit.core.common import Graph, BaseNode
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.graph.base_node import WeightAttrT
from model_compression_toolkit.core.common.graph.memory_graph.compute_graph_max_cut import compute_graph_max_cut
from model_compression_toolkit.core.common.graph.memory_graph.cut import Cut
from model_compression_toolkit.core.common.graph.memory_graph.memory_graph import MemoryGraph
from model_compression_toolkit.core.common.graph.virtual_activation_weights_node import VirtualActivationWeightsNode, \
    VirtualSplitWeightsNode
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization import \
    RUTarget, ResourceUtilization
from model_compression_toolkit.core.common.quantization.node_quantization_config import NodeWeightsQuantizationConfig, \
    NodeActivationQuantizationConfig, BaseNodeQuantizationConfig
from model_compression_toolkit.core.common.substitutions.virtual_activation_weights_composition import \
    get_input_activation_if_composable


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
    bytes: float

    def __add__(self, other: 'Utilization') -> 'Utilization':
        """ Add another Utilization object. """
        return Utilization(self.size + other.size, self.bytes + other.bytes)

    def __radd__(self, other: Literal[0]):
        """ Right add is only supported with 0 to allow the sum operator (with the default start_value=0) """
        if other != 0:
            raise ValueError('radd is only supported with 0')
        return self

    def __gt__(self, other: 'Utilization'):
        """ Greater than operator by bytes. Needed for max. """
        return self.bytes > other.bytes

    def __lt__(self, other: 'Utilization'):
        """ Less than operator by bytes. Needed for min. """
        return self.bytes < other.bytes


NodeName = str
ActivationQCfgPerNode = Dict[NodeName, NodeActivationQuantizationConfig]
WeightsQCfgPerNode = Dict[NodeName, NodeWeightsQuantizationConfig]
DetailedMem = Dict[Union[BaseNode, Cut], float]


class ResourceUtilizationCalculator:
    """ Resource utilization calculator. """

    _bitwidth_mode_fn = {
        BitwidthMode.QMaxBit: max,
        BitwidthMode.QMinBit: min,
    }

    unexpected_qc_error = 'Custom quantization configuration is not expected for non-custom bit mode.'
    unexpected_qc_nodes_error = 'Custom quantization configuration contains unexpected node names.'

    def __init__(self, graph: Graph, fw_impl: FrameworkImplementation, fw_info: FrameworkInfo):
        self.graph = graph
        self.fw_impl = fw_impl
        self.fw_info = fw_info

        # Currently we go over the full graph even if utilization won't be requested for all nodes.
        # We could fill the cache on the fly only for requested nodes, but it's probably negligible.
        self._act_tensors_size = {}
        self._params_cnt = {}
        for n in graph.nodes:
            self._act_tensors_size[n.name] = n.get_total_output_params()
            if n.weights:
                self._params_cnt[n.name] = {k: v.size for k, v in n.weights.items()}
        self._cuts: Optional[Dict[Cut, List[BaseNode]]] = None
        self._nodes_names = set(n.name for n in graph.nodes)

    @property
    def cuts(self) -> Dict[Cut, List[BaseNode]]:
        """ Compute if needed and return graph cuts and their memory element nodes. """
        if self._cuts is None:
            cuts = self._compute_cuts()
            if cuts is None:    # pragma: no cover
                raise RuntimeError("Failed to calculate activation memory cuts for graph.")
            cuts = [cut for cut in cuts if cut.mem_elements.elements]
            # cache cuts nodes for future use, so do not filter by target
            self._cuts = {cut: [self.graph.find_node_by_name(m.node_name)[0] for m in cut.mem_elements.elements]
                          for cut in cuts}
        return self._cuts

    def compute_resource_utilization(self,
                                     target_criterion: TargetInclusionCriterion,
                                     bitwidth_mode: BitwidthMode,
                                     act_qcs: Optional[ActivationQCfgPerNode] = None,
                                     w_qcs: Optional[WeightsQCfgPerNode] = None,
                                     ru_targets: Iterable[RUTarget] = None,
                                     allow_unused_qcs: bool = False,
                                     return_detailed=False) \
            -> Union[ResourceUtilization, Tuple[ResourceUtilization, Dict[RUTarget, DetailedMem]]]:
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
            allow_unused_qcs: by default, if custom quantization configs are passed, but are not going to be used for
              any of the requested targets, an error is raised. To disable the validation, pass True.
            return_detailed: whether to return an additional dictionary with detailed utilization per element.

        Returns:
            Resource utilization object, or a tuple of resource utilization object and a dict containing detailed
            memory utilization per ru target: for weights and bops targets - bytes per node,
                                              for activations and total targets - bytes per cut.
        """
        ru_targets = set(ru_targets) if ru_targets else set(RUTarget)

        self._validate_custom_qcs(act_qcs, bitwidth_mode)
        self._validate_custom_qcs(w_qcs, bitwidth_mode)

        if w_qcs and not {RUTarget.WEIGHTS, RUTarget.TOTAL, RUTarget.BOPS}.intersection(ru_targets):
            if not allow_unused_qcs:
                raise ValueError('Weight configuration passed but no relevant ru_targets requested.')
            w_qcs = None

        if act_qcs and not {RUTarget.ACTIVATION, RUTarget.TOTAL, RUTarget.BOPS}.intersection(ru_targets):
            if not allow_unused_qcs:
                raise ValueError('Activation configuration passed but no relevant ru_targets requested.')
            act_qcs = None

        w_total, w_per_node = None, None
        if {RUTarget.WEIGHTS, RUTarget.TOTAL}.intersection(ru_targets):
            w_total, w_per_node, _ = self.compute_weights_utilization(target_criterion, bitwidth_mode, w_qcs)

        a_total, a_per_cut = None, None
        if {RUTarget.ACTIVATION, RUTarget.TOTAL}.intersection(ru_targets):
            a_total, a_per_cut, _ = self.compute_activations_utilization(target_criterion, bitwidth_mode, act_qcs)

        ru = ResourceUtilization()
        detailed = {}
        if RUTarget.WEIGHTS in ru_targets:
            ru.weights_memory = w_total
            if return_detailed:
                detailed[RUTarget.WEIGHTS] = {n: u.bytes for n, u in w_per_node.items()}
        if RUTarget.ACTIVATION in ru_targets:
            ru.activation_memory = a_total
            if return_detailed:
                detailed[RUTarget.ACTIVATION] = {cut: u.bytes for cut, u in a_per_cut.items()}
        if RUTarget.TOTAL in ru_targets:
            ru.total_memory = w_total + a_total
            if return_detailed:
                detailed[RUTarget.TOTAL] = {cut: u.bytes + w_total for cut, u in a_per_cut.items()}
        if RUTarget.BOPS in ru_targets:
            ru.bops, bops_per_node = self.compute_bops(target_criterion, bitwidth_mode, act_qcs=act_qcs, w_qcs=w_qcs)
            if return_detailed:
                detailed[RUTarget.BOPS] = bops_per_node

        assert ru.get_restricted_targets() == set(ru_targets), \
            'Mismatch between the number of requested and computed metrics'

        if return_detailed:
            return ru, detailed

        return ru

    def compute_weights_utilization(self,
                                    target_criterion: TargetInclusionCriterion,
                                    bitwidth_mode: BitwidthMode,
                                    w_qcs: Optional[WeightsQCfgPerNode] = None) \
            -> Tuple[float, Dict[NodeName, Utilization], Dict[NodeName, Dict[str, Utilization]]]:
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
        self._validate_custom_qcs(w_qcs, bitwidth_mode)

        node_attrs = self._collect_target_nodes_w_attrs(target_criterion, include_reused=False)

        util_per_node: Dict[NodeName, Utilization] = {}
        util_per_node_per_weight = {}
        for n in self._topo_sort(list(node_attrs.keys())):
            w_qc = w_qcs.get(n.name) if w_qcs else None
            node_weights_util, per_weight_util = self.compute_node_weights_utilization(n, node_attrs[n],
                                                                                       bitwidth_mode, w_qc)
            util_per_node[n.name] = node_weights_util
            util_per_node_per_weight[n.name] = per_weight_util

        total_util = sum(util_per_node.values()) if util_per_node else Utilization(0, 0)
        return total_util.bytes, util_per_node, util_per_node_per_weight

    def compute_node_weights_utilization(self,
                                         n: BaseNode,
                                         target_criterion: Union[TargetInclusionCriterion, List[str]],
                                         bitwidth_mode: BitwidthMode,
                                         qc: Optional[NodeWeightsQuantizationConfig] = None)\
            -> Tuple[Utilization, Dict[str, Utilization]]:
        """
        Compute resource utilization for weights of a node.

        Args:
            n: node.
            target_criterion: criterion to include weights for computation, or explicit attributes list (full names).
            bitwidth_mode: bit-width mode for the computation.
            qc: custom weights quantization configuration. Should be provided for custom bit mode only.
              In custom mode, must provide configuration for all configurable weights. For non-configurable
              weights, if not provided, the default configuration will be extracted from the node.

        Returns:
            - Node's total weights utilization.
            - Detailed per weight attribute utilization.
        """
        if qc:
            if bitwidth_mode != BitwidthMode.QCustom:
                raise ValueError(self.unexpected_qc_error)
            if set(qc.all_weight_attrs) - set(n.get_node_weights_attributes()):
                raise ValueError(f'Custom configuration contains unexpected weight attrs {qc.all_weight_attrs} for '
                                 f'node {n} containing weight attrs {n.get_node_weights_attributes()}.')

        # If target criterion is passed, weights_attrs may return empty, that's fine.
        # However, if an explicit list is passed, it must be non-empty.
        if isinstance(target_criterion, TargetInclusionCriterion):
            weight_attrs = self._get_target_weight_attrs(n, target_criterion)
        else:
            weight_attrs = target_criterion
            if not weight_attrs:
                raise ValueError('Explicit list of attributes to compute cannot be empty.')

        attr_util = {}
        for attr in weight_attrs:
            size = self._params_cnt[n.name][attr]
            nbits = self._get_weight_nbits(n, attr, bitwidth_mode, qc)
            bytes_ = size * nbits / 8
            attr_util[attr] = Utilization(size, bytes_)

        total_weights: Utilization = sum(attr_util.values()) if attr_util else Utilization(0, 0)
        return total_weights, attr_util

    def compute_activations_utilization(self,
                                        target_criterion: TargetInclusionCriterion,
                                        bitwidth_mode: BitwidthMode,
                                        act_qcs: Optional[ActivationQCfgPerNode] = None):
        """
        Compute total activations utilization in the graph.

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
        return self.compute_activation_utilization_by_cut(target_criterion, bitwidth_mode, act_qcs)

    def compute_activation_utilization_by_cut(self,
                                              target_criterion: TargetInclusionCriterion,
                                              bitwidth_mode: BitwidthMode,
                                              act_qcs: Optional[ActivationQCfgPerNode] = None) \
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
        self._validate_custom_qcs(act_qcs, bitwidth_mode)

        graph_target_nodes = self._get_target_activation_nodes(target_criterion, include_reused=True)
        # if there are no target activations in the graph, don't waste time looking for cuts
        if not graph_target_nodes:
            return 0, {}, {}

        util_per_cut: Dict[Cut, Utilization] = {}
        util_per_cut_per_node = defaultdict(dict)
        for cut in self.cuts:
            cut_target_nodes = self._get_cut_target_nodes(cut, target_criterion)
            if not cut_target_nodes:
                continue
            for n in cut_target_nodes:
                qc = act_qcs.get(n.name) if act_qcs else None
                util_per_cut_per_node[cut][n.name] = self.compute_node_activation_tensor_utilization(n, target_criterion,
                                                                                                     bitwidth_mode, qc)
            util_per_cut[cut] = sum(util_per_cut_per_node[cut].values())    # type: ignore

        total_util = max(util_per_cut.values())
        return total_util.bytes, util_per_cut, util_per_cut_per_node

    def compute_activation_tensors_utilization(self,
                                               target_criterion: TargetInclusionCriterion,
                                               bitwidth_mode: BitwidthMode,
                                               act_qcs: Optional[ActivationQCfgPerNode] = None,
                                               include_reused=False) \
            -> Tuple[float, Dict[NodeName, Utilization]]:
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
        self._validate_custom_qcs(act_qcs, bitwidth_mode)

        nodes = self._get_target_activation_nodes(target_criterion, include_reused=include_reused)

        util_per_node: Dict[NodeName, Utilization] = {}
        for n in self._topo_sort(nodes):
            qc = act_qcs.get(n.name) if act_qcs else None
            util = self.compute_node_activation_tensor_utilization(n, None, bitwidth_mode, qc)
            util_per_node[n.name] = util

        total_util = max(util_per_node.values()).bytes if util_per_node else 0
        return total_util, util_per_node

    def compute_node_activation_tensor_utilization(self,
                                                   n: BaseNode,
                                                   target_criterion: Optional[TargetInclusionCriterion],
                                                   bitwidth_mode: BitwidthMode,
                                                   qc: Optional[NodeActivationQuantizationConfig] = None) -> Utilization:
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
        if qc and bitwidth_mode != BitwidthMode.QCustom:
            raise ValueError(self.unexpected_qc_error)

        if target_criterion:
            # only check whether the node meets the criterion
            nodes = self._get_target_activation_nodes(target_criterion=target_criterion, include_reused=True, nodes=[n])
            if not nodes:
                return Utilization(0, 0)

        size = self._act_tensors_size[n.name]
        nbits = self._get_activation_nbits(n, bitwidth_mode, qc)
        bytes_ = size * nbits / 8
        return Utilization(size, bytes_)

    def compute_bops(self,
                     target_criterion: TargetInclusionCriterion,
                     bitwidth_mode: BitwidthMode,
                     act_qcs: Optional[ActivationQCfgPerNode] = None,
                     w_qcs: Optional[WeightsQCfgPerNode] = None) \
            -> Tuple[int, Dict[NodeName, int]]:
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
        self._validate_custom_qcs(act_qcs, bitwidth_mode)
        self._validate_custom_qcs(w_qcs, bitwidth_mode)

        nodes_bops = {}
        for n in self.graph.get_topo_sorted_nodes():
            w_qc = w_qcs.get(n.name) if w_qcs else None
            bops = self.compute_node_bops(n, target_criterion, bitwidth_mode, act_qcs=act_qcs, w_qc=w_qc)
            if bops:
                nodes_bops[n.name] = bops

        return sum(nodes_bops.values()), nodes_bops

    def compute_node_bops(self,
                          n: BaseNode,
                          target_criterion: Optional[TargetInclusionCriterion],
                          bitwidth_mode: BitwidthMode,
                          act_qcs: Optional[ActivationQCfgPerNode] = None,
                          w_qc: Optional[NodeWeightsQuantizationConfig] = None) -> Union[float, int]:
        """
        Compute Bit Operations of a node.

        Args:
            n: node.
            target_criterion: criterion to include nodes for computation.
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
        if target_criterion is None:
            target_criterion = TargetInclusionCriterion.Any
        if target_criterion not in [TargetInclusionCriterion.AnyQuantized, TargetInclusionCriterion.Any]:
            raise ValueError('BOPS computation is supported only for Any and AnyQuantized targets.')

        self._validate_custom_qcs(act_qcs, bitwidth_mode)
        self._validate_custom_qcs(w_qc, bitwidth_mode)

        if isinstance(n, VirtualSplitWeightsNode):
            # Virtual weights node can only be present if it couldn't be merged into VirtualActivationWeightsNode.
            # This means that during MP search we cannot compute bops for all A/W nbits combinations. To prevent
            # inconsistencies we ignore such nodes for bops computation.
            return 0

        # Fetch the original weights node for mac computation (VirtualActivationWeightsNode input/output shapes are
        # based on the activation original node, not weights original node)
        orig_w_node = n
        if isinstance(n, VirtualActivationWeightsNode):
            orig_w_node = n.original_weights_node
            if isinstance(orig_w_node, VirtualSplitWeightsNode):
                orig_w_node = orig_w_node.origin_node

        # check if the node has kernel
        kernel_attrs = self.fw_info.get_kernel_op_attributes(n.type)
        if len(kernel_attrs) > 1:  # pragma: no cover
            raise NotImplementedError('Multiple kernel attributes are not supported for BOPS computation.')
        if not kernel_attrs or not kernel_attrs[0]:
            return 0

        kernel_attr = kernel_attrs[0]
        node_mac = self.fw_impl.get_node_mac_operations(orig_w_node, self.fw_info)
        if node_mac == 0:
            return node_mac

        # find the activation node from which to get quantization info and for which to look in custom configuration
        if isinstance(n, VirtualActivationWeightsNode):
            # we don't need the original node (and cannot use it for custom configuration anyway)
            a_node = n
        else:
            # if we are running on the original (non-virtual) graph, we only compute bops if it would be computed in an
            # equivalent virtual graph for consistency.
            a_node = get_input_activation_if_composable(self.graph, n, warn=False)
            if a_node is None:
                return 0

        if (target_criterion == TargetInclusionCriterion.AnyQuantized and
                not (a_node.is_activation_quantization_enabled() or n.is_weights_quantization_enabled(kernel_attr))):
            return 0

        act_qc = act_qcs.get(a_node.name) if act_qcs else None
        a_nbits = self._get_activation_nbits(a_node, bitwidth_mode, act_qc)
        w_nbits = self._get_weight_nbits(n, kernel_attr, bitwidth_mode, w_qc)
        node_bops = a_nbits * w_nbits * node_mac
        return node_bops

    def _compute_cuts(self):
        """ Compute activation cuts of the graph. """
        memory_graph = MemoryGraph(deepcopy(self.graph))
        _, _, cuts = compute_graph_max_cut(memory_graph)
        return cuts

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

    def _collect_target_nodes_w_attrs(self,
                                      target_criterion: TargetInclusionCriterion,
                                      include_reused: bool) -> Dict[BaseNode, List[WeightAttrT]]:
        """
        Collect nodes and their weight attributes to include in weights utilization computation.

        Args:
            target_criterion: criterion to include weights for computation.
            include_reused: whether to include reused nodes.

        Returns:
            A mapping from nodes to their weights attributes.
        """
        nodes_attrs = {n: attrs for n in self.graph.nodes
                       if (attrs := self._get_target_weight_attrs(n, target_criterion))
                       and (include_reused or not n.reuse)}
        return nodes_attrs

    @staticmethod
    def _get_target_weight_attrs(n: BaseNode, target_criterion: TargetInclusionCriterion) -> List[str]:
        """
        Collect weight attributes of a node per criterion.

        Args:
            n: node.
            target_criterion: selection criterion.

        Returns:
            Selected weight attributes names.
        """
        # weight_attrs are the full names in the layer, e.g. 'conv2d_1/kernel:0' (or an integer for positional attrs)
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
            nodes: nodes to sort. Allowed to be empty.

        Returns:
            Nodes in topological order.
        """
        if not nodes:
            return list(nodes)

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
        elif target_criterion != TargetInclusionCriterion.Any:    # pragma: no cover
            raise ValueError(f'Unknown {target_criterion}.')
        if not include_reused:
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
            assert bitwidth_mode == BitwidthMode.QCustom
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
            if len(qcs) != 1:
                raise ValueError(f'Could not retrieve the activation quantization candidate for node {n} '
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
        assert not (w_qc and bitwidth_mode != BitwidthMode.QCustom)
        if w_qc and w_qc.has_attribute_config(w_attr):
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
            if len(w_qcs) != 1:
                raise ValueError(f'Could not retrieve the quantization candidate for attr {w_attr} of node {n} '
                                 f'as it has {len(w_qcs)}!=1 unique candidates.')
            return w_qcs[0].weights_n_bits

        raise ValueError(f'Unknown mode {bitwidth_mode.name}')    # pragma: no cover

    def _validate_custom_qcs(self,
                             qcs: Union[BaseNodeQuantizationConfig, Dict[NodeName, BaseNodeQuantizationConfig]],
                             bitwidth_mode: BitwidthMode):
        """
        Validate custom quantization configuration.

        Args:
            qcs: either a mapping from nodes names to quantization configuration, or just a quantization configuration.
            bitwidth_mode: bit mode.

        Raises:
            ValueError: if validation fails.

        """
        if qcs is None:
            return

        if bitwidth_mode != BitwidthMode.QCustom:
            raise ValueError(self.unexpected_qc_error)

        if isinstance(qcs, (NodeActivationQuantizationConfig, NodeWeightsQuantizationConfig)):
            return

        unknown_nodes = set(qcs.keys()) - self._nodes_names
        if unknown_nodes:
            raise ValueError(self.unexpected_qc_nodes_error, unknown_nodes)

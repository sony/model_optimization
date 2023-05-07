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

from typing import Dict, Any, Tuple

from model_compression_toolkit.core import FrameworkInfo
from model_compression_toolkit.constants import VIRTUAL_ACTIVATION_WEIGHTS_NODE_PREFIX, \
    VIRTUAL_WEIGHTS_SUFFIX, VIRTUAL_ACTIVATION_SUFFIX, FLOAT_BITWIDTH

from model_compression_toolkit.core.common.graph.base_node import BaseNode
import numpy as np

from model_compression_toolkit.core.common.quantization.candidate_node_quantization_config import \
    CandidateNodeQuantizationConfig


class VirtualSplitNode(BaseNode):
    """
    A class that represents a node that was split from a kernel node (node with weights).
    """

    def __init__(self, origin_node: BaseNode):
        """
        Init a VirtualSplitNode object.

        Args:
            origin_node: The original node from which the new node was split.
        """

        super().__init__(origin_node.name,
                         origin_node.framework_attr,
                         origin_node.input_shape,
                         origin_node.output_shape,
                         origin_node.weights,
                         origin_node.layer_class,
                         origin_node.reuse,
                         origin_node.reuse_group,
                         origin_node.quantization_attr,
                         origin_node.has_activation)

        self.origin_node = origin_node


class VirtualSplitWeightsNode(VirtualSplitNode):
    """
    A class that represents a node that was split from a kernel node (node with weights) and holds the weights of
    the original node. This node contains the original node's weights and the relevant weights candidate quantization
    config.
    """

    def __init__(self, origin_node: BaseNode):
        """
        Init a VirtualSplitWeightsNode object.

        Args:
            origin_node: The original node from which the new node was split.
        """

        super().__init__(origin_node)

        self.name = origin_node.name + VIRTUAL_WEIGHTS_SUFFIX

        self.candidates_quantization_cfg = origin_node.get_unique_weights_candidates()
        for c in self.candidates_quantization_cfg:
            c.activation_quantization_cfg.enable_activation_quantization = False
            c.activation_quantization_cfg.activation_n_bits = FLOAT_BITWIDTH


class VirtualSplitActivationNode(VirtualSplitNode):
    """
    A class that represents a node that was split from a kernel node (node with weights) and holds the activation
    operation of the original node. This node basically does not apply any operation and only holds the relevant
    activation candidate quantization config.
    """

    def __init__(self, origin_node: BaseNode, activation_class: type, fw_attr: dict):
        """
        Init a VirtualSplitActivationNode object.

        Args:
            origin_node: The original node from which the new node was split.
        """

        super().__init__(origin_node)

        self.name = origin_node.name + VIRTUAL_ACTIVATION_SUFFIX
        self.framework_attr = fw_attr
        self.prior_info = origin_node.prior_info
        self.input_shape = origin_node.output_shape  # the kernel output is the activation input
        self.weights = {}
        self.layer_class = activation_class

        self.candidates_quantization_cfg = origin_node.get_unique_activation_candidates()
        for c in self.candidates_quantization_cfg:
            c.weights_quantization_cfg.enable_weights_quantization = False
            c.weights_quantization_cfg.weights_n_bits = FLOAT_BITWIDTH


class VirtualActivationWeightsNode(BaseNode):
    """
    A node that represents a composition of pair of sequential activation node and weights (kernel) node.
    This structure is used for mixed-precision search with bit-operation KPI.
    The node's candidates are the cartesian product of both nodes' candidates.

    Important: note that not like regular BaseNode or FunctionalNode, in VirtualActivationWeightsNode the activation
    candidates config refer to the quantization config of the activation that precedes the linear operation! instead of
    the output of the linear operation.
    It is ok, since this node is not meant to be used in a graph for creating an actual model, but only a virtual
    representation of the model's graph only for allowing to compute the bit-operations KPI in mixed-precision.
    """

    def __init__(self,
                 act_node: BaseNode,
                 weights_node: BaseNode,
                 name: str,
                 framework_attr: Dict[str, Any],
                 input_shape: Tuple[Any],
                 output_shape: Tuple[Any],
                 weights: Dict[str, np.ndarray],
                 layer_class: type,
                 reuse: bool = False,
                 reuse_group: str = None,
                 quantization_attr: Dict[str, Any] = None,
                 has_activation: bool = True,
                 **kwargs):
        """
        Init a VirtualActivationWeightsNode object.

        Args:
            act_node: The original activation node.
            weights_node: The original weights node.
            name: Node's name
            framework_attr: Framework attributes the layer had which the node holds.
            input_shape: Input tensor shape of the node.
            output_shape: Input tensor shape of the node.
            weights: Dictionary from a variable name to the weights with that name in the layer the node represents.
            layer_class: Class path of the layer this node represents.
            reuse: Whether this node was duplicated and represents a reused layer.
            reuse_group: Name of group of nodes from the same reused layer.
            quantization_attr: Attributes the node holds regarding how it should be quantized.
            has_activation: Whether the node has activations that we might want to quantize.
            **kwargs: Additional arguments that can be passed but are not used (allows to init the object with an
                existing node's __dict__).

        """

        super().__init__(name,
                         framework_attr,
                         input_shape,
                         output_shape,
                         weights,
                         layer_class,
                         reuse,
                         reuse_group,
                         quantization_attr,
                         has_activation)

        self.name = f"{VIRTUAL_ACTIVATION_WEIGHTS_NODE_PREFIX}_{act_node.name}_{weights_node.name}"

        self.original_activation_node = act_node
        self.original_weights_node = weights_node

        v_candidates = []
        for c_a in act_node.candidates_quantization_cfg:
            for c_w in weights_node.candidates_quantization_cfg:
                composed_candidate = CandidateNodeQuantizationConfig(activation_quantization_cfg=c_a.activation_quantization_cfg,
                                                                     weights_quantization_cfg=c_w.weights_quantization_cfg)
                v_candidates.append(composed_candidate)

        # sorting the candidates by weights number of bits first and then by activation number of bits (reversed order)
        v_candidates.sort(key=lambda c: (c.weights_quantization_cfg.weights_n_bits,
                                         c.activation_quantization_cfg.activation_n_bits), reverse=True)

        self.candidates_quantization_cfg = v_candidates

    def get_bops_count(self, fw_impl: Any, fw_info: FrameworkInfo, candidate_idx: int) -> float:
        """
        Computes the composed node's (edge) bit-operation count.

        Args:
            fw_impl: A FrameworkImplementation object with framework specific methods.
            fw_info: A FrameworkInfo object with framework specific information,
            candidate_idx: The index of the node's quantization candidate configuration.

        Returns: The BOPS count of the composed node.

        """
        node_mac = fw_impl.get_node_mac_operations(self.original_weights_node, fw_info)
        candidate = self.candidates_quantization_cfg[candidate_idx]
        weights_bit = candidate.weights_quantization_cfg.weights_n_bits if \
            candidate.weights_quantization_cfg.enable_weights_quantization else FLOAT_BITWIDTH
        activation_bit = candidate.activation_quantization_cfg.activation_n_bits if \
            candidate.activation_quantization_cfg.enable_activation_quantization else FLOAT_BITWIDTH
        node_bops = weights_bit * activation_bit * node_mac
        return node_bops

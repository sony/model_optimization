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
import uuid

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

    def __init__(self, origin_node: BaseNode, kernel_attr: str):
        """
        Init a VirtualSplitWeightsNode object.

        Args:
            origin_node: The original node from which the new node was split.
            kernel_attr: The name of the kernel attribute of the original node.
        """

        super().__init__(origin_node)

        self.name = origin_node.name + VIRTUAL_WEIGHTS_SUFFIX
        # Virtual weights node is created only to be absorbed into virtual composed node right away.
        # However, in some cases composition is impossible and virtual weights node can remain in the graph.
        # In such case it messes up resource utilization computation, specifically activation cuts. In order to minimize
        # the impact, we preserve the behavior of the original node wrt activation (shape and quantization),
        # so that prev - virtualW cut is identical to prev-origin_node. Only the cut virtualW-virtualA will be different
        # from the original graph, so in the worst case the utilization will be higher in virtual graph.
        # This should guarantee that the utilization of the original graph does not exceed the requested target.
        self.candidates_quantization_cfg = origin_node.candidates_quantization_cfg


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
    This structure is used for mixed-precision search with bit-operation constraint.
    The node's candidates are the cartesian product of both nodes' candidates.

    Important: note that not like regular BaseNode or FunctionalNode, in VirtualActivationWeightsNode the activation
    candidates config refer to the quantization config of the activation that precedes the linear operation! instead of
    the output of the linear operation.
    It is ok, since this node is not meant to be used in a graph for creating an actual model, but only a virtual
    representation of the model's graph only for allowing to compute the bit-operations constraint in mixed-precision.
    """

    def __init__(self,
                 act_node: BaseNode,
                 weights_node: BaseNode,
                 fw_info: FrameworkInfo):
        """
        Init a VirtualActivationWeightsNode object.

        Args:
            act_node: The original activation node.
            weights_node: The original weights node.
            fw_info: A FrameworkInfo object with framework specific information.
        """
        # Validate weights node
        kernel_attrs = fw_info.get_kernel_op_attributes(weights_node.type)
        assert len(kernel_attrs) == 1 and kernel_attrs[0] is not None, f'Expected exactly one kernel attr, {kernel_attrs}'
        kernel_attr = kernel_attrs[0]
        conf_weights = [attr for attr in weights_node.weights if weights_node.is_configurable_weight(attr)]
        if len(conf_weights) > 1 or len(conf_weights) == 1 and not weights_node.is_configurable_weight(kernel_attr):
            raise NotImplementedError(f'Only kernel weight can be configurable. Got configurable {conf_weights}.')

        weights = weights_node.weights.copy()
        act_node_w_rename = {}
        if act_node.weights:
            if not fw_info.get_kernel_op_attributes(act_node)[0] is None:
                raise NotImplementedError(f'Node {act_node} with kernel cannot be used as activation for '
                                          f'VirtualActivationWeightsNode.')
            if act_node.has_any_configurable_weight():
                raise NotImplementedError(f'Node {act_node} with a configurable weight cannot be used as activation for '
                                          'VirtualActivationWeightsNode.')
            # combine weights from activation and weights
            for w_id, w in act_node.weights.items():
                if w_id not in weights and not (isinstance(w_id, str) and kernel_attr in w_id):
                    weights[w_id] = w
                    continue
                # if same identifier is used as in weight nodes (or contains the kernel substring), generate a new
                # unique id. If positional, generate a new (and clearly made up) index.
                # This only serves for resource utilization computation so in theory this shouldn't matter, as long as
                # quantization config dict keys are updated accordingly.
                uniq_id = uuid.uuid4().hex[:8] if isinstance(w_id, str) else (100 + w_id)
                assert uniq_id not in weights
                act_node_w_rename[w_id] = uniq_id
                weights[uniq_id] = w

        name = f"{VIRTUAL_ACTIVATION_WEIGHTS_NODE_PREFIX}_{act_node.name}_{weights_node.name}"
        super().__init__(name,
                         framework_attr=weights_node.framework_attr,
                         input_shape=act_node.input_shape,
                         output_shape=act_node.output_shape,
                         weights=weights,
                         layer_class=weights_node.layer_class,
                         reuse=weights_node.reuse,
                         reuse_group=weights_node.reuse_group,
                         quantization_attr=weights_node.quantization_attr,
                         has_activation=False)

        self.original_activation_node = act_node
        self.original_weights_node = weights_node

        v_candidates = []
        weights_candidates_quantization_cfg = weights_node.get_unique_weights_candidates(kernel_attr)
        for c_a in act_node.candidates_quantization_cfg:
            for c_w in weights_candidates_quantization_cfg:
                composed_candidate = CandidateNodeQuantizationConfig(activation_quantization_cfg=c_a.activation_quantization_cfg,
                                                                     weights_quantization_cfg=c_w.weights_quantization_cfg)
                if act_node.weights:
                    # add non-kernel weights cfg from activation node to the composed node's weights cfg
                    composed_candidate.weights_quantization_cfg.attributes_config_mapping.update(
                        {act_node_w_rename.get(k, k): v
                         for k, v in c_a.weights_quantization_cfg.attributes_config_mapping.items()}
                    )
                    composed_candidate.weights_quantization_cfg.pos_attributes_config_mapping.update(
                        {act_node_w_rename.get(k, k): v
                         for k, v in c_a.weights_quantization_cfg.pos_attributes_config_mapping.items()}
                    )
                v_candidates.append(composed_candidate)

        # sorting the candidates by weights number of bits first and then by activation number of bits (reversed order)
        v_candidates.sort(key=lambda c: (c.weights_quantization_cfg.get_attr_config(kernel_attr).weights_n_bits,
                                         c.activation_quantization_cfg.activation_n_bits), reverse=True)

        self.candidates_quantization_cfg = v_candidates

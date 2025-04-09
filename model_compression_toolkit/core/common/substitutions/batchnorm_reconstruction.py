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


import copy
from typing import Callable

import numpy as np

from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.quantization.quantization_config import QuantizationConfig
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common.quantization.node_quantization_config import WeightsAttrQuantizationConfig, \
    ActivationQuantizationMode
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.core.common.graph.base_node import BaseNode
from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher
from mct_quantizers import QuantizationMethod
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import AttributeQuantizationConfig


class BatchNormalizationReconstruction(common.BaseSubstitution):
    """
    Reconstruct BatchNormalization after linear layers.
    """
    def __init__(self,
                 source_node: NodeOperationMatcher,
                 create_bn_node: Callable,
                 gamma_str: str,
                 beta_str: str,
                 moving_mean_str: str,
                 moving_variance_str: str,
                 epsilon_val: float):
        """
        Matches: Source (linear) nodes.

        Args:
            source_node: Node matcher for linear type nodes.
            create_bn_node: Function for creating BN nodes.
            gamma_str: The framework specific attribute name of the batch norm layer's gamma parameter.
            beta_str: The framework specific attribute name of the batch norm layer's beta parameter.
            moving_mean_str: The framework specific attribute name of the batch norm layer's moving mean parameter.
            moving_variance_str: The framework specific attribute name of the batch norm layer's moving
            variance parameter.
            epsilon_val: The framework specific attribute value of the batch norm layer's epsilon parameter.
        """
        super().__init__(matcher_instance=source_node)
        self.create_bn_node = create_bn_node
        self.gamma_str = gamma_str
        self.beta_str = beta_str
        self.moving_mean_str = moving_mean_str
        self.moving_variance_str = moving_variance_str
        self.epsilon_val = epsilon_val

    def substitute(self,
                   graph: Graph,
                   source_node: BaseNode) -> Graph:
        """
        Reconstruct BatchNormalization after linear layers.

        Args:
            graph: Graph we apply the substitution on.
            source_node: linear type node.

        Returns:
            Graph after applying the substitution.
        """

        num_nodes_before_substitution = len(graph.nodes)
        num_edges_before_substitution = len(graph.edges)

        # If the linear operator is part of a reused group (it is the "base" node, or a reused node),
        # we should skip the substitution.
        if source_node.is_reused():
            for qc in source_node.candidates_quantization_cfg:
                qc.weights_quantization_cfg.weights_second_moment_correction = False
            return graph

        # We apply only on nodes with folded BatchNormalization.
        if source_node.prior_info.std_output is None or source_node.prior_info.mean_output is None:
            for qc in source_node.candidates_quantization_cfg:
                qc.weights_quantization_cfg.weights_second_moment_correction = False
            return graph

        # This feature disabled for models with weights quantization method of Power of 2
        for qc in source_node.candidates_quantization_cfg:
            # this feature is relevant only for layers with kernel op
            kernel_attr = graph.fw_info.get_kernel_op_attributes(source_node.type)
            if kernel_attr is None:
                Logger.error(f"Can't preform BatchNorm reconstruction on a node {source_node.name} without a kernel op.")
            if (qc.weights_quantization_cfg.get_attr_config(kernel_attr[0]).weights_quantization_method
                    == QuantizationMethod.POWER_OF_TWO):
                Logger.warning("Second moment statistics correction feature disabled for models with weights "
                               "quantization method of Power of 2")
                for qc_inner in source_node.candidates_quantization_cfg:
                    qc_inner.weights_quantization_cfg.weights_second_moment_correction = False
                return graph

        eps = self.epsilon_val

        original_gamma = source_node.prior_info.std_output
        beta = source_node.prior_info.mean_output
        moving_mean = beta
        moving_var = np.power(original_gamma, 2)
        gamma = np.sqrt(moving_var + eps)

        bn_node_weights = {self.gamma_str: gamma,
                           self.beta_str: beta,
                           self.moving_mean_str: moving_mean,
                           self.moving_variance_str: moving_var}

        bn_node = self.create_bn_node(source_node, bn_node_weights)

        bn_node.prior_info = copy.deepcopy(source_node.prior_info)

        bn_node.candidates_quantization_cfg = copy.deepcopy(source_node.candidates_quantization_cfg)

        for qc in bn_node.candidates_quantization_cfg:
            qc.activation_quantization_cfg.quant_mode = ActivationQuantizationMode.NO_QUANT
            for attr in bn_node.get_node_weights_attributes():
                if qc.weights_quantization_cfg.has_attribute_config(attr):
                    # we only create a BN layer to collect statistics, so we don't need to quantize anything,
                    # but we do need to add the BN attributes to the reconstructed node.
                    qc.weights_quantization_cfg.get_attr_config(attr).enable_weights_quantization = False
                else:
                    # setting a "dummy" attribute configuration with disabled quantization.
                    # TODO: once enabling BN attributes quantization, need to figure out if thie
                    #  reconstructed node BN attributes need to be quantized and how.
                    qc.weights_quantization_cfg.set_attr_config(attr,
                                                                WeightsAttrQuantizationConfig(
                                                                    QuantizationConfig(),
                                                                    AttributeQuantizationConfig(
                                                                        enable_weights_quantization=False)))

        # Check if the source node was part of a fusion. If so, there are two cases:
        # either this is no longer a fusion, and the fusion info should be updated by removing
        # the current info, or this creates a new fusion and the old pattern should be
        # replaced with the new one.
        fi = graph.fusing_info
        fused_op = fi.get_fused_node_name(source_node.name)
        if fused_op:
            fused_nodes = list(fi.get_fused_nodes(fused_op))
            assert source_node in fused_nodes
            fused_nodes.insert(fused_nodes.index(source_node)+1, bn_node)
            fi.remove_fused_operation(fused_op)
            if fi.is_nodes_eligible_to_be_fused(fused_nodes):
                op_id = fi.generate_fused_op_id(fused_nodes)
                fi.add_fused_operation(op_id, tuple(fused_nodes))

        graph.reconnect_out_edges(current_node=source_node, new_node=bn_node)
        graph.replace_output_node(current_node=source_node, new_node=bn_node)
        graph.add_node_with_in_edges(bn_node, [source_node])

        assert len(graph.nodes) - num_nodes_before_substitution == 1
        assert len(graph.edges) - num_edges_before_substitution == 1

        return graph

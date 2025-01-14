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
import numpy as np
from typing import Tuple, Callable

from model_compression_toolkit.core import common
from model_compression_toolkit.core.common.graph.base_graph import Graph
from model_compression_toolkit.core.common.graph.graph_matchers import EdgeMatcher, NodeOperationMatcher
from model_compression_toolkit.core.common.graph.base_node import BaseNode
from mct_quantizers import QuantizationMethod
from model_compression_toolkit.constants import THRESHOLD, RANGE_MIN, RANGE_MAX
from model_compression_toolkit.logger import Logger


class BatchNormalizationRefusing(common.BaseSubstitution):
    """
    Re-fuse BatchNormalization into preceding linear layers.
    """

    def __init__(self,
                 source_node: NodeOperationMatcher,
                 bn_node: NodeOperationMatcher,
                 update_kernel_for_bn_refusing_fn: Callable,
                 kernel_str: str,
                 bias_str: str,
                 gamma_str: str,
                 beta_str: str,
                 moving_mean_str: str,
                 moving_variance_str: str,
                 epsilon_str: str,
                 use_bias: str,
                 layer_name_str: str):
        """
        Matches: Conv node (source node) to Batch Normalization (bn node).

        Args:
            source_node: Node matcher for convolution type nodes.
            bn_node: Node matcher for batch normalization nodes.
            update_kernel_for_bn_refusing_fn: Function for updating the convolution kernel
            with the batch normalization weights
            kernel_str: The framework specific attribute name of the convolution layer's weight/kernel.
            bias_str: The framework specific attribute name of the convolution layer's bias.
            gamma_str: The framework specific attribute name of the batch norm layer's gamma parameter.
            beta_str: The framework specific attribute name of the batch norm layer's beta parameter.
            moving_mean_str: The framework specific attribute name of the batch norm layer's moving mean parameter.
            moving_variance_str: The framework specific attribute name of the batch norm layer's moving variance
            parameter.
            epsilon_str: The framework specific attribute name of the batch norm layer's epsilon parameter.
            use_bias: The framework specific attribute name of the convolution layer's bias flag.
            layer_name_str: The framework specific attribute name of layer's name.
        """
        super().__init__(matcher_instance=EdgeMatcher(source_node, bn_node))
        self.update_kernel_for_bn_refusing_fn = update_kernel_for_bn_refusing_fn
        self.kernel_str = kernel_str
        self.bias_str = bias_str
        self.gamma_str = gamma_str
        self.beta_str = beta_str
        self.moving_mean_str = moving_mean_str
        self.moving_variance_str = moving_variance_str
        self.epsilon_str = epsilon_str
        self.use_bias = use_bias
        self.layer_name_str = layer_name_str
        self.layer_name_suffix = '_refused'

    def substitute(self,
                   graph: Graph,
                   edge_nodes: Tuple[BaseNode, BaseNode]) -> Graph:
        """
        Re-fuse BatchNormalization into preceding linear layers.

        Args:
            graph: Graph we apply the substitution on.
            edge_nodes: Tuple of tow nodes (linear op and batchnorm node).

        Returns:
            Graph after applying the substitution.
        """

        num_nodes_before_substitution = len(graph.nodes)
        num_edges_before_substitution = len(graph.edges)

        source_node = edge_nodes[0]

        # We apply only on nodes with reconstructed BatchNormalization.
        if not source_node.final_weights_quantization_cfg.weights_second_moment_correction:
            return graph

        # If the linear operator is part of a reused group (it is the "base" node, or a reused node),
        # we should skip the substitution.
        if source_node.is_reused():
            Logger.critical("BN folding substitution cannot proceed if the linear operator is part of a reused group.")  # pragma: no cover

        bn_node = edge_nodes[1]

        if len(graph.get_next_nodes(source_node)) > 1 or len(graph.get_prev_nodes(bn_node)) > 1:
            Logger.critical(
                "BN folding substitution cannot proceed if the linear operator has multiple outputs or the BN layer has multiple inputs.")  # pragma: no cover

        kernel = source_node.get_weights_by_keys(self.kernel_str)
        bias = source_node.get_weights_by_keys(self.bias_str)
        gamma = bn_node.get_weights_by_keys(self.gamma_str)
        beta = bn_node.get_weights_by_keys(self.beta_str)
        moving_mean = bn_node.get_weights_by_keys(self.moving_mean_str)
        moving_variance = bn_node.get_weights_by_keys(self.moving_variance_str)
        eps = bn_node.framework_attr[self.epsilon_str]

        weights_scale = gamma / np.sqrt(moving_variance + eps)
        bias = beta + (bias - moving_mean) * weights_scale

        if not isinstance(weights_scale, np.ndarray):
            weights_scale = weights_scale.numpy()

        kernel, kernel_name = self.update_kernel_for_bn_refusing_fn(source_node, kernel, weights_scale)

        framework_attr = copy.copy(source_node.framework_attr)
        framework_attr[self.use_bias] = True
        if self.layer_name_str is not None:
            framework_attr[self.layer_name_str] = source_node.name + self.layer_name_suffix

        weights_dict = {kernel_name: kernel,
                        self.bias_str: bias}

        conv_bn = copy.deepcopy(source_node)
        conv_bn_name = source_node.name + self.layer_name_suffix
        conv_bn.name = conv_bn_name
        conv_bn.framework_attr = framework_attr
        conv_bn.weights = weights_dict

        graph.add_node(conv_bn)
        graph.reconnect_out_edges(current_node=bn_node, new_node=conv_bn)
        graph.reconnect_in_edges(current_node=source_node, new_node=conv_bn)

        graph.replace_output_node(current_node=bn_node, new_node=conv_bn)

        conv_bn.prior_info = bn_node.prior_info
        in_stats = graph.get_in_stats_collector(source_node)
        out_stats = graph.get_out_stats_collector(source_node)
        graph.set_out_stats_collector_to_node(conv_bn, out_stats)
        graph.node_to_in_stats_collector.update({conv_bn: in_stats})

        graph.remove_edge(source_node, bn_node)
        graph.remove_node(bn_node)
        graph.remove_node(source_node)

        self._calc_weights_quantization_params(conv_bn, weights_scale, graph.fw_info)

        assert num_nodes_before_substitution - len(graph.nodes) == 1
        assert num_edges_before_substitution - len(graph.edges) == 1
        return graph

    def _calc_weights_quantization_params(self,
                                          conv_bn: BaseNode,
                                          weights_scale: np.ndarray,
                                          fw_info):
        """
        Update node weights quantization params.
        Args:
            conv_bn: Convolution node to update the weights quantization params.
            weights_scale: Weight scale factor in which to multiply the conv node's weight.
            fw_info: FrameworkInfo object with information about the specific framework's model
        """
        # Conv layer is ensured to have a kernel attribute
        kernel_attr = fw_info.get_kernel_op_attributes(conv_bn.type)[0]
        conv_bn_kernel_cfg = conv_bn.final_weights_quantization_cfg.get_attr_config(kernel_attr)
        # In case of SYMMETRIC weight quantization method, we update the threshold by weights_scale
        if conv_bn_kernel_cfg.weights_quantization_method == QuantizationMethod.SYMMETRIC:
            original_threshold = conv_bn_kernel_cfg.weights_quantization_params[THRESHOLD]
            corr_dict = copy.deepcopy(conv_bn_kernel_cfg.weights_quantization_params)
            corr_threshold, _ = self.update_kernel_for_bn_refusing_fn(conv_bn, original_threshold, weights_scale)
            corr_dict[THRESHOLD] = corr_threshold
            conv_bn_kernel_cfg.set_weights_quantization_param(corr_dict)

        # In case of UNIFORM weight quantization method, we update the range_min, range_max by weights_scale
        elif conv_bn_kernel_cfg.weights_quantization_method == QuantizationMethod.UNIFORM:
            corr_dict = copy.deepcopy(conv_bn_kernel_cfg.weights_quantization_params)
            original_range_min = conv_bn_kernel_cfg.weights_quantization_params[RANGE_MIN]
            corr_range_min, _ = self.update_kernel_for_bn_refusing_fn(conv_bn, original_range_min, weights_scale)
            original_range_max = conv_bn_kernel_cfg.weights_quantization_params[RANGE_MAX]
            corr_range_max, _ = self.update_kernel_for_bn_refusing_fn(conv_bn, original_range_max, weights_scale)
            corr_dict[RANGE_MIN] = corr_range_min
            corr_dict[RANGE_MAX] = corr_range_max
            conv_bn_kernel_cfg.set_weights_quantization_param(corr_dict)

        else:
            Logger.critical("Second moment statistics correction feature is not supported for weights quantization methods other than 'SYMMETRIC' and 'UNIFORM'.")  # pragma: no cover

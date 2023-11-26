# Copyright 2021 Sony Semiconductor Israel, Inc. All rights reserved.
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


from tensorflow.keras.layers import SeparableConv2D, Conv2D, DepthwiseConv2D

from model_compression_toolkit.core import common
from model_compression_toolkit.core.common.graph.base_graph import Graph
from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher
from model_compression_toolkit.core.common.graph.base_node import BaseNode
from model_compression_toolkit.core.keras.constants import KERNEL, DEPTHWISE_KERNEL, BIAS, KERNEL_SIZE, PADDING, \
    STRIDES, USE_BIAS, LINEAR, ACTIVATION, TRAINABLE, FILTERS, PAD_VALID, GROUPS

POINTWISE_KERNEL = 'pointwise_kernel'
DEPTH_MULTIPLIER = 'depth_multiplier'
DATA_FORMAT = 'data_format'
DILATION_RATE = 'dilation_rate'
DEPTHWISE_INITIALIZER = 'depthwise_initializer'
DEPTHWISE_REGULARIZER = 'depthwise_regularizer'
DEPTHWISE_CONSTRAINT = 'depthwise_constraint'
BIAS_INITIALIZER = 'bias_initializer'
BIAS_REGULARIZER = 'bias_regularizer'
BIAS_CONSTRAINT = 'bias_constraint'
ACTIVITY_REGULARIZER = 'activity_regularizer'
KERNEL_INITIALIZER = 'kernel_initializer'
SEPARABLE_PW_KERNEL_INITIALIZER = 'pointwise_initializer'
KERNEL_REGULARIZER = 'kernel_regularizer'
SEPARABLE_PW_KERNEL_REGULARIZER = 'pointwise_regularizer'
KERNEL_CONSTRAINT = 'kernel_constraint'
SEPARABLE_PW_KERNEL_CONSTRAINT = 'pointwise_constraint'


class SeparableConvDecomposition(common.BaseSubstitution):
    """
    Remove a SeparableConv2D node from the graph,
    and replace it with two equivalent nodes: DepthwiseConv2D and Conv2D.
    """

    def __init__(self):
        """
        Matches SeparableConv2D nodes.
        """
        separable_node = NodeOperationMatcher(SeparableConv2D)
        super().__init__(matcher_instance=separable_node)

    def substitute(self,
                   graph: Graph,
                   separable_node: BaseNode) -> Graph:
        """
        Remove a SeparableConv2D node from the graph, and replace it with two equivalent nodes: DepthwiseConv2D
        and Conv2D. The SeparableConv2D attributes are split to relevant attributes for each node.

        Args:
            graph: Graph we apply the substitution on.
            separable_node: Separable node to replace with a depthwise and pointwise nodes.

        Returns:
            Graph after applying the substitution.
        """

        dw_kernel = separable_node.get_weights_by_keys(DEPTHWISE_KERNEL)
        pw_kernel = separable_node.get_weights_by_keys(POINTWISE_KERNEL)
        pw_bias = separable_node.get_weights_by_keys(BIAS)

        dw_weights_dict = {DEPTHWISE_KERNEL: dw_kernel}
        pw_weights_dict = {KERNEL: pw_kernel,
                           BIAS: pw_bias}

        # Split separable node attributes into relevant attributes for each of the new nodes.
        # List of dw attributes that should take from separable as they are.
        dw_attr_list = [KERNEL_SIZE, STRIDES, PADDING, DEPTH_MULTIPLIER, DATA_FORMAT, DILATION_RATE,
                        DEPTHWISE_INITIALIZER, DEPTHWISE_REGULARIZER, DEPTHWISE_CONSTRAINT, TRAINABLE]

        dw_framework_attr = {attr: separable_node.framework_attr[attr] for attr in dw_attr_list}
        dw_framework_attr.update({ACTIVATION: LINEAR,
                                  USE_BIAS: False})

        # List of pw attributes that should take from separable as they are.
        pw_attr_list = [FILTERS, DATA_FORMAT, DILATION_RATE, ACTIVATION, USE_BIAS, BIAS_CONSTRAINT,
                        BIAS_INITIALIZER, BIAS_REGULARIZER, TRAINABLE, ACTIVITY_REGULARIZER, GROUPS]

        pw_framework_attr = {attr: separable_node.framework_attr[attr] for attr in pw_attr_list}

        # Use more attributes that are not taken as are
        pw_framework_attr.update({KERNEL_SIZE: (1, 1),
                                  STRIDES: (1, 1),
                                  PADDING: PAD_VALID,
                                  KERNEL_INITIALIZER: separable_node.framework_attr[SEPARABLE_PW_KERNEL_INITIALIZER],
                                  KERNEL_REGULARIZER: separable_node.framework_attr[SEPARABLE_PW_KERNEL_REGULARIZER],
                                  KERNEL_CONSTRAINT: separable_node.framework_attr[SEPARABLE_PW_KERNEL_CONSTRAINT]})

        # two new nodes will replace the separable node: depthwise and pointwise convolutions
        dw_layer_class = DepthwiseConv2D
        pw_layer_class = Conv2D

        # compute input/outpus shapes of new nodes
        dw_output_shape = tuple(dw_layer_class(**dw_framework_attr).compute_output_shape(separable_node.input_shape))
        pw_input_shape = dw_output_shape

        # If the SeparableConv2D is reused, we need to keep the depthwise node as reused as well,
        # so we keep the names convention with adding the suffix of "_reuse_X".
        dw_node_name = separable_node.name + '_dw' if not separable_node.reuse else '_'.join(separable_node.name.split('_')[:-2]) + '_dw_' + '_'.join(separable_node.name.split('_')[-2:])
        reuse_group = separable_node.reuse_group if not separable_node.reuse_group else separable_node.reuse_group + '_dw'


        # create new nodes
        dw_node = common.graph.BaseNode(dw_node_name,
                                        dw_framework_attr,
                                        separable_node.input_shape,
                                        dw_output_shape,
                                        dw_weights_dict,
                                        dw_layer_class,
                                        reuse=separable_node.reuse,
                                        reuse_group=reuse_group)

        # If the SeparableConv2D is reused, we need to keep the pointwise node as reused as well,
        # so we keep the names convention with adding the suffix of "_reuse_X".
        pw_node_name = separable_node.name + '_pw' if not separable_node.reuse else '_'.join(separable_node.name.split('_')[:-2]) + '_pw_' + '_'.join(separable_node.name.split('_')[-2:])
        reuse_group = separable_node.reuse_group if not separable_node.reuse_group else separable_node.reuse_group + '_pw'

        pw_node = common.graph.BaseNode(pw_node_name,
                                        pw_framework_attr,
                                        pw_input_shape,
                                        separable_node.output_shape,
                                        pw_weights_dict,
                                        pw_layer_class,
                                        reuse=separable_node.reuse,
                                        reuse_group=reuse_group)

        graph.add_node(dw_node)
        graph.add_node(pw_node)
        graph.add_edge(dw_node,
                       pw_node,
                       source_index=0,
                       sink_index=0)

        # connect edges to new nodes
        graph.reconnect_in_edges(current_node=separable_node, new_node=dw_node)
        graph.reconnect_out_edges(current_node=separable_node, new_node=pw_node)

        # if separable node was one the model output nodes, we need to update the graph outputs according to the
        # substitution.
        graph.replace_output_node(current_node=separable_node, new_node=pw_node)
        graph.remove_node(separable_node)

        return graph

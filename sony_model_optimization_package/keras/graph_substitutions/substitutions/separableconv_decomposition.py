# ===============================================================================
# Copyright (c) 2021, Sony Semiconductors Israel, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ===============================================================================


from tensorflow.keras.layers import SeparableConv2D, Conv2D, DepthwiseConv2D

from sony_model_optimization_package import common
from sony_model_optimization_package.common.graph.base_graph import Graph
from sony_model_optimization_package.common.graph.graph_matchers import NodeOperationMatcher
from sony_model_optimization_package.common.graph.node import Node
from sony_model_optimization_package.keras.constants import KERNEL, DEPTHWISE_KERNEL, BIAS, KERNEL_SIZE, PADDING, \
    STRIDES, USE_BIAS, LINEAR, ACTIVATION, TRAINABLE, FILTERS, PAD_VALID

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
                   separable_node: Node) -> Graph:
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

        dw_weights_dict = {KERNEL: dw_kernel}
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
                        BIAS_INITIALIZER, BIAS_REGULARIZER, TRAINABLE, ACTIVITY_REGULARIZER]

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

        # create new nodes
        dw_node = common.graph.Node(separable_node.name + '_dw',
                                    dw_framework_attr,
                                    separable_node.input_shape,
                                    dw_output_shape,
                                    dw_weights_dict,
                                    dw_layer_class)

        pw_node = common.graph.Node(separable_node.name + '_pw',
                                    pw_framework_attr,
                                    pw_input_shape,
                                    separable_node.output_shape,
                                    pw_weights_dict,
                                    pw_layer_class)

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

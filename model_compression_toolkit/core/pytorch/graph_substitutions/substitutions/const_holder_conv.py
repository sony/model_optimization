# Copyright 2022 Sony Semiconductors Israel, Inc. All rights reserved.
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
from torch.nn import Conv2d, ConvTranspose2d
from torch.nn.functional import conv2d, conv_transpose2d
from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher, EdgeMatcher
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common.graph.base_graph import Graph
from model_compression_toolkit.core.common.graph.base_node import BaseNode
from model_compression_toolkit.core.pytorch.reader.graph_builders import ConstantHolder
from model_compression_toolkit.core.pytorch.constants import IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, KERNEL, BIAS, CONSTANT
from model_compression_toolkit.core.common import FrameworkInfo

class ConstantHolderConv(common.BaseSubstitution):
    """
    Find "ConstantHolder" followed by conv layer to substitute a new single layer with weights if needed
    """

    def __init__(self, fw_info: FrameworkInfo):
        """
        Matches: 'ConstantHolder' followed by conv layer
        """
        first_node = NodeOperationMatcher(ConstantHolder)
        second_node = NodeOperationMatcher(conv2d) | NodeOperationMatcher(conv_transpose2d)
        self.fw_info = fw_info
        super().__init__(matcher_instance=EdgeMatcher(first_node, second_node))

    def substitute(self,
                   graph: Graph,
                   nodes: BaseNode) -> Graph:
        """
        Substitute ConstantHolder and conv/linear layer with single layer
        Args:
            graph: Graph we apply the substitution on.
            nodes: nodes that match the pattern in the substitution init.

        Returns:
            Graph after applying the substitution.
        """
        first_node = nodes[0] # constant holder node
        second_node = nodes[1] # convolution node

        # Check if there is a connection
        if graph.get_edge_data(first_node,second_node) is None:
            return graph # skip substitution

        # Set new layer
        if second_node.type == conv2d:
            NewLayer = Conv2d
        elif second_node.type == conv_transpose2d:
            NewLayer = ConvTranspose2d
        else:
            return graph # skip substitution

        out_channel_index, in_channel_index = self.fw_info.kernel_channels_mapping.get(NewLayer)

        # Check if there is a bias node
        bias_node = None
        prev_nodes = graph.get_prev_nodes(second_node)
        prev_nodes = list(filter(lambda prev_node: prev_node.type == ConstantHolder, prev_nodes))
        if len(prev_nodes) == 2:
            bias_node = prev_nodes[1]

        # Create new node of layer convolution
        weights = first_node.get_weights_by_keys(CONSTANT)
        framework_attr = second_node.framework_attr
        framework_attr.update({OUT_CHANNELS: weights.shape[out_channel_index]})
        framework_attr.update({IN_CHANNELS: weights.shape[in_channel_index]})
        framework_attr.update({KERNEL_SIZE: weights.shape[2:]})

        new_node = BaseNode(name=second_node.name,
                            framework_attr=framework_attr,
                            input_shape=second_node.input_shape,
                            output_shape=second_node.output_shape,
                            weights={KERNEL: weights} if bias_node is None else {KERNEL: weights, BIAS: bias_node.get_weights_by_keys(CONSTANT)},
                            layer_class=NewLayer,
                            has_activation=second_node.has_activation)
        graph.add_node(new_node)
        if bias_node is not None:
            graph.remove_edge(bias_node, second_node)
            if len(graph.get_next_nodes(bias_node)) == 0:
                graph.remove_node(bias_node)
        graph.remove_edge(first_node, second_node)
        if len(graph.get_next_nodes(first_node)) == 0:
            graph.remove_node(first_node)
        graph.reconnect_out_edges(current_node=second_node, new_node=new_node)
        graph.reconnect_in_edges(current_node=second_node, new_node=new_node)
        graph.replace_output_node(current_node=second_node, new_node=new_node)
        graph.remove_node(second_node)

        return graph


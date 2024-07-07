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
from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common.graph.base_graph import Graph
from model_compression_toolkit.core.common.graph.base_node import BaseNode
from model_compression_toolkit.core.common.graph.functional_node import FunctionalNode
from model_compression_toolkit.core.pytorch.constants import IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, KERNEL, BIAS
from model_compression_toolkit.core.common import FrameworkInfo


class FunctionalConvSubstitution(common.BaseSubstitution):
    """
    Substitute functional convolutions with Layers
    """
    def __init__(self, fw_info: FrameworkInfo):
        """
        Matches a functional conv node
        """
        func_node = NodeOperationMatcher(conv2d) | NodeOperationMatcher(conv_transpose2d)
        self.fw_info = fw_info
        super().__init__(matcher_instance=func_node)

    def substitute(self,
                   graph: Graph,
                   func_node: FunctionalNode) -> Graph:
        """
        Substitute functional and conv/linear layer with torch layer
        Args:
            graph: Graph we apply the substitution on.
            func_node: node that match the pattern in the substitution init.

        Returns:
            Graph after applying the substitution.
        """
        # Set new layer
        if func_node.is_match_type(conv2d):
            new_layer = Conv2d
        elif func_node.is_match_type(conv_transpose2d):
            new_layer = ConvTranspose2d
        else:
            Logger.critical(f'Substitution filter mismatch. Layer {func_node.type}. Must be {type(Conv2d)} or {type(ConvTranspose2d)}.')  # pragma: no cover

        out_channel_index, in_channel_index = self.fw_info.kernel_channels_mapping.get(new_layer)

        # Create new node of layer convolution
        if 1 not in func_node.weights:
            Logger.critical(f'Weight input missing for node {func_node.name}.')  # pragma: no cover
        # Extract index of kernel and bias according to tensor_input_allocs if they were input as kwargs. If
        # they were input as args, use their fixed positions.
        weight_index = func_node.tensor_input_allocs.index(KERNEL) if KERNEL in func_node.tensor_input_allocs else 1
        bias_index = func_node.tensor_input_allocs.index(BIAS) if BIAS in func_node.tensor_input_allocs else 2
        if weight_index not in func_node.weights:
            Logger.critical(f'Mismatch between tensor_input_allocs and weight index in node {func_node.name}.')  # pragma: no cover
        weight = func_node.weights[weight_index]
        bias = func_node.weights.get(bias_index)
        framework_attr = func_node.op_call_kwargs
        framework_attr.update({OUT_CHANNELS: weight.shape[out_channel_index]})
        framework_attr.update({IN_CHANNELS: weight.shape[in_channel_index]})
        framework_attr.update({KERNEL_SIZE: weight.shape[2:]})

        new_node = BaseNode(name=func_node.name,
                            framework_attr=framework_attr,
                            input_shape=func_node.input_shape,
                            output_shape=func_node.output_shape,
                            weights={KERNEL: weight} if bias is None else {KERNEL: weight, BIAS: bias},
                            layer_class=new_layer,
                            has_activation=func_node.has_activation)
        graph.add_node(new_node)
        graph.reconnect_out_edges(current_node=func_node, new_node=new_node)
        graph.reconnect_in_edges(current_node=func_node, new_node=new_node)
        graph.replace_output_node(current_node=func_node, new_node=new_node)
        graph.remove_node(func_node)

        return graph


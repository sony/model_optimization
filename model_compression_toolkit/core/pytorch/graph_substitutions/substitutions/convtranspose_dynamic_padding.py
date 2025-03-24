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
from typing import Tuple
import torch.nn as nn
import torch
from model_compression_toolkit.core.pytorch.constants import OUTPUT_PADDING
from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common import BaseNode, Graph
from model_compression_toolkit.logger import Logger


class ConvtransposeDynamicPadding(common.BaseSubstitution):
    """
    Replace output_padding of nn.ConvTranspose2d to align dynamic output_size input.
    In case there is a dynamic output_size in ConvTranspose2d forward function, we recalculate the
    output_padding here according to node.output_shape (which is equal to the dynamic output_size if existed).
    """

    def __init__(self):
        """
        Matches: nn.ConvTranspose2d
        """
        convtr_node = NodeOperationMatcher(nn.ConvTranspose2d)
        super().__init__(matcher_instance=convtr_node)


    def calc_dynamic_output_size(self, node: BaseNode) -> Tuple[int]:
        """
        Calc the output padding to support dunamic output_size of nn.ConvTranspose2d
        Args:
            node: node to calculate output padding

        Returns:
            corrected output padding
        """
        convtr = nn.ConvTranspose2d(**node.framework_attr)
        num_spatial_dims = 2
        output_padding = convtr._output_padding(torch.randn(size=node.input_shape[0]),
                                                node.output_shape[0],
                                                convtr.stride,
                                                convtr.padding,
                                                convtr.kernel_size,
                                                num_spatial_dims,
                                                convtr.dilation)
        return tuple(output_padding)


    def substitute(self,
                   graph: Graph,
                   node: BaseNode) -> Graph:
        """
        Substitute nn.ConvTranspose2d with corrected output_padding for cases of dynamic output_size
        Args:
            graph: Graph we apply the substitution on.
            node: node that match the pattern in the substitution init.

        Returns:
            Graph after applying the substitution.
        """

        if not node.reuse:
            output_padding = self.calc_dynamic_output_size(node)
            node.framework_attr.update({OUTPUT_PADDING: output_padding})
        return graph

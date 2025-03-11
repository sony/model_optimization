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
    Replace output_padding of nn.convtranspose to align dynamic output_size input
    """

    def __init__(self):
        """
        Matches: functional batch_norm
        """
        convtr_node = NodeOperationMatcher(nn.ConvTranspose2d)
        super().__init__(matcher_instance=convtr_node)


    def calc_dynamic_output_size(self, node: BaseNode) -> Tuple[int]:
        """
        Calc the output padding to support dunamic output_size of nn.ConvTranspose2d
        Args:
            node: node to calculate output padding

        Returns:
            correct output padding
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
        Substitute functional.batch_norm and its inputs with BatchNorm2d.
        Args:
            graph: Graph we apply the substitution on.
            node: node that match the pattern in the substitution init.

        Returns:
            Graph after applying the substitution.
        """
        # Check that output is only contain a single tensor
        if len(node.output_shape) > 1:
            Logger.critical('Output to nn.ConvTranspose2d should be a single tensor but got more than one.')  # pragma: no cover
        output_padding = self.calc_dynamic_output_size(node)
        node.framework_attr.update({OUTPUT_PADDING: output_padding})
        return graph

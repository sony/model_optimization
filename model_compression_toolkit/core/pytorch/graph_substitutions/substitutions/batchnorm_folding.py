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
import numpy as np
from torch.nn import BatchNorm2d, Conv2d, ConvTranspose2d

from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher
from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.common.substitutions.batchnorm_folding import BatchNormalizationFolding
from model_compression_toolkit.core.pytorch.constants import KERNEL, BIAS, GAMMA, BETA, MOVING_MEAN, MOVING_VARIANCE, \
    EPSILON, USE_BIAS


def batchnorm_folding_node_matchers():
    """
    Function generates matchers for matching:
    (Conv2d, ConvTranspose2d)-> BatchNorm2d.

    Returns:
        Matcher for batch norm nodes, and source nodes.
    """
    bn_node = NodeOperationMatcher(BatchNorm2d)
    source_node = NodeOperationMatcher(Conv2d) | \
                  NodeOperationMatcher(ConvTranspose2d)
    return bn_node, source_node


def update_kernel_for_bn_folding_fn(conv_node: BaseNode,
                                    kernel: np.ndarray,
                                    weights_scale: np.ndarray):
    """
    Args:
        conv_node: Convolution node to update the weight/kernel.
        kernel: The Convolution node's weight
        weights_scale: Weight scale factor in which to multiply the conv node's weight.

    Returns:
        The modified convolution node's weight/kernel/
    """
    return kernel * weights_scale[:, None, None, None], KERNEL


def pytorch_batchnorm_folding() -> BatchNormalizationFolding:
    """

    Returns:
        A BatchNormalizationFolding initialized for Pytorch models.
    """
    bn_node, source_node = batchnorm_folding_node_matchers()
    return BatchNormalizationFolding(source_node,
                                      bn_node,
                                      update_kernel_for_bn_folding_fn,
                                      KERNEL,
                                      BIAS,
                                      GAMMA,
                                      BETA,
                                      MOVING_MEAN,
                                      MOVING_VARIANCE,
                                      EPSILON,
                                      USE_BIAS,
                                      layer_name_str=None,  # torch.nn.Modules don't have an attribute 'name'
                                      )

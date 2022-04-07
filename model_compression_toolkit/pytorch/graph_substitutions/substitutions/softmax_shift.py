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

from torch.nn import Softmax, Linear
from torch.nn.functional import softmax

from model_compression_toolkit.common.graph.graph_matchers import NodeOperationMatcher, WalkMatcher
from model_compression_toolkit.common.substitutions.softmax_shift import SoftmaxShift
from model_compression_toolkit.pytorch.constants import BIAS


def softmax_shift_matcher():
    """
    Matches: (Linear, Softmax)
    """
    activation_node = NodeOperationMatcher(Softmax) | NodeOperationMatcher(softmax)
    source_node = NodeOperationMatcher(Linear)
    return WalkMatcher([source_node, activation_node])


def pytorch_softmax_shift() -> SoftmaxShift:
    """
    Shift the layer before Softmax activation.
    Returns:
        Graph after applying the substitution for Pytorch models.
    """
    nodes = softmax_shift_matcher()
    return SoftmaxShift(nodes,
                        BIAS)

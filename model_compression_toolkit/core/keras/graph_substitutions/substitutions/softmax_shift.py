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

from tensorflow.keras.layers import Softmax, Dense

from model_compression_toolkit.core.common.graph.graph_matchers import NodeFrameworkAttrMatcher, NodeOperationMatcher, \
    WalkMatcher
from model_compression_toolkit.core.common.substitutions.softmax_shift import SoftmaxShift
from model_compression_toolkit.core.keras.constants import BIAS, SOFTMAX, ACTIVATION


def softmax_shift_matcher():
    """
    Matches: (Dense, Softmax)
    """
    activation_node = NodeFrameworkAttrMatcher(ACTIVATION, SOFTMAX) | NodeOperationMatcher(Softmax)
    source_node = NodeOperationMatcher(Dense)
    return WalkMatcher([source_node, activation_node])


def keras_softmax_shift() -> SoftmaxShift:
    """
    Shift the layer before Softmax activation.
    Returns:
        Graph after applying the substitution for Keras models.
    """
    nodes = softmax_shift_matcher()
    return SoftmaxShift(nodes,
                        BIAS)

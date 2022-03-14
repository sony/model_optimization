# Copyright 2021 Sony Semiconductors Israel, Inc. All rights reserved.
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


from tensorflow.keras.layers import ReLU

from model_compression_toolkit import common
from model_compression_toolkit.common import Graph, BaseNode
from model_compression_toolkit.common.graph.graph_matchers import NodeOperationMatcher,NodeFrameworkAttrMatcher
from model_compression_toolkit.keras.constants import RELU_MAX_VALUE
from model_compression_toolkit.common.constants import THRESHOLD

MATCHER = NodeOperationMatcher(ReLU) & NodeFrameworkAttrMatcher(RELU_MAX_VALUE, None).logic_not()


class RemoveReLUUpperBound(common.BaseSubstitution):
    """
    Remove ReLU upper bound if its activation threshold bounds it anyway at
    the same value.
    """


    def __init__(self):
        """
        Initialize a RemoveReLUUpperBound object.
        """
        super().__init__(matcher_instance=MATCHER)

    def substitute(self,
                   graph: Graph,
                   node: BaseNode) -> Graph:
        """
        Remove ReLU upper bound if its activation threshold bounds it anyway at
        the same value.

        Args:
            graph: Graph we apply the substitution on.
            node: Node to remove its bound.

        Returns:
            Graph after applying the substitution.
        """
        if node.final_activation_quantization_cfg and \
                node.final_activation_quantization_cfg.activation_quantization_params.get(THRESHOLD) == \
                node.framework_attr.get(RELU_MAX_VALUE):
            node.framework_attr[RELU_MAX_VALUE] = None
            common.Logger.info(f'Removing upper bound of {node.name}. Threshold and upper bound are equal.')
        return graph

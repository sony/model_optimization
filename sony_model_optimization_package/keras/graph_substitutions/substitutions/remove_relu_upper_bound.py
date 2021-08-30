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


from tensorflow.keras.layers import ReLU

from sony_model_optimization_package import common
from sony_model_optimization_package.common import Graph, Node
from sony_model_optimization_package.common.graph.graph_matchers import NodeOperationMatcher,NodeFrameworkAttrMatcher
from sony_model_optimization_package.keras.constants import RELU_MAX_VALUE
from sony_model_optimization_package.common.constants import THRESHOLD

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
                   node: Node) -> Graph:
        """
        Remove ReLU upper bound if its activation threshold bounds it anyway at
        the same value.

        Args:
            graph: Graph we apply the substitution on.
            node: Node to remove its bound.

        Returns:
            Graph after applying the substitution.
        """
        if node.activation_quantization_cfg.activation_quantization_params.get(THRESHOLD) == node.framework_attr.get(RELU_MAX_VALUE):
            node.framework_attr[RELU_MAX_VALUE] = None
            common.Logger.info(f'Removing upper bound of {node.name}. Threshold and upper bound are equal.')
        return graph

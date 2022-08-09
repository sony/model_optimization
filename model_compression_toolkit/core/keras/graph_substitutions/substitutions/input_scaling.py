# Copyright 2021 Sony Semiconductor Israel, Inc. All rights reserved.
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


from tensorflow.keras.layers import InputLayer, Dense, DepthwiseConv2D, Conv2D, Conv2DTranspose, ZeroPadding2D
from typing import List

from model_compression_toolkit.core import common
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common.graph.base_graph import Graph
from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher, EdgeMatcher, WalkMatcher
from model_compression_toolkit.core.common.graph.base_node import BaseNode
from model_compression_toolkit.core.common.quantization.quantization_config import QuantizationConfig
from model_compression_toolkit.core.common.constants import THRESHOLD
from model_compression_toolkit.core.keras.constants import KERNEL

input_node = NodeOperationMatcher(InputLayer)
zeropad_node = NodeOperationMatcher(ZeroPadding2D)
op2d_node = NodeOperationMatcher(Dense) | \
            NodeOperationMatcher(Conv2D) | \
            NodeOperationMatcher(DepthwiseConv2D) | \
            NodeOperationMatcher(Conv2DTranspose)

INPUT_MATCHER = WalkMatcher([input_node, op2d_node])
INPUT_MATCHER_WITH_PAD = WalkMatcher([input_node, zeropad_node, op2d_node])


class BaseInputScaling(common.BaseSubstitution):
    """
    General scale activation threshold for input layers, if they are followed by linear nodes. We first
    scale their thresholds to a constrained threshold, and then fix it by scaling the linear op weights
    correspondingly.
    The matcher instance of type WalkMatcher may include intermediate nodes that don't affect scaling
    (such as ZeroPadding), but only the first and last nodes are used for scaling
    """

    def __init__(self,
                 matcher_instance):
        """
        Matches: InputLayer -> (optional nodes) -> (Dense,Conv2D,DepthwiseConv2D,Conv2DTranspose)
        note: the optional nodes are nodes that don't affect the scaling (such as ZeroPadding)

        Create a substitution using different params which may affect the way this substitution is made.
        The substitution is looking for edges in the graph which are input layers connected to linear layers.
        Args:
            matcher_instance: matcher instance of type WalkMatcher

        """
        super().__init__(matcher_instance=matcher_instance)

    def substitute(self,
                   graph: Graph,
                   nodes_list: List[BaseNode]) -> Graph:
        """
        Scale activation threshold for input layers, if they are followed by linear nodes. We first
        scale their thresholds to a constrained threshold, and then fix it by scaling the linear op weights
        correspondingly.

        Args:
            graph: Graph to apply the substitution on.
            edge_nodes: Edge (tuple of nodes) that matches the pattern the substitution can be applied on.

        Returns:
            Graph after applying the substitution.
        """

        input_layer = nodes_list[0]
        linear_layer = nodes_list[-1]

        if not input_layer.is_all_activation_candidates_equal():
            raise Exception("Input scaling is not supported for more than one activation quantization configuration "
                            "candidate")

        # all candidates have same activation config, so taking the first candidate for calculations
        threshold = input_layer.candidates_quantization_cfg[0].activation_quantization_cfg.activation_quantization_params.get(THRESHOLD)

        if threshold is None:
            return graph

        min_value, max_value = graph.get_out_stats_collector(input_layer).get_min_max_values()
        threshold_float = max(abs(min_value), max_value)

        if threshold > threshold_float:
            scale_factor = threshold_float / threshold
            graph.user_info.set_input_scale(1 / scale_factor)

            w1_fixed = linear_layer.get_weights_by_keys(KERNEL) * scale_factor
            linear_layer.set_weights_by_keys(KERNEL, w1_fixed)

            graph.scale_stats_collector(input_layer, 1 / scale_factor)

            # After scaling weights may have different thresholds so it needs to be recalculated
            for nqc in linear_layer.candidates_quantization_cfg:
                nqc.weights_quantization_cfg.calculate_and_set_weights_params(w1_fixed)

        return graph


class InputScaling(BaseInputScaling):
    """
    Substitution extends BaseInputScaling to the case of Input-->Linear
    """

    def __init__(self):
        """
        Initialize a ScaleEqualization object.
        """

        super().__init__(matcher_instance=INPUT_MATCHER)


class InputScalingWithPad(BaseInputScaling):
    """
    Substitution extends BaseInputScaling to the case of Input-->ZeroPadding-->Linear
    """

    def __init__(self):
        """
        Initialize a ScaleEqualization object.
        """

        super().__init__(matcher_instance=INPUT_MATCHER_WITH_PAD)
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


import tensorflow as tf
from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper
from typing import Tuple, List
from tensorflow.keras.layers import Layer

from network_optimization_package.common.graph.base_graph import Graph
from network_optimization_package.common.graph.node import Node
from network_optimization_package.keras.constants import USE_BIAS
from network_optimization_package.keras.quantizer.configs import ActivationQuantizeConfigKD, WeightQuantizeConfigKD
from network_optimization_package.common.framework_info import FrameworkInfo
from tensorflow.keras.models import Model


def get_compare_points(input_graph: Graph) -> Tuple[List[Node], List[str]]:
    """
    Create a list of nodes in a graph where we collect their output statistics, and a corresponding list
    of their names for tensors comparison purposes.
    Args:
        input_graph: Graph to get its points to compare.

    Returns:
        A list of nodes in a graph, and a list of the their names.
    """
    compare_points = []
    compare_points_name = []
    for n in input_graph.nodes():
        tensors = input_graph.get_out_stats_collector(n)
        if (not isinstance(tensors, list)) and tensors.require_collection():
            compare_points.append(n)
            compare_points_name.append(n.name)
    return compare_points, compare_points_name


def get_trainable_parameters(fxp_model: Model,
                             fw_info: FrameworkInfo,
                             add_bias: bool = False) -> List[tf.Variable]:
    """
    Get trainable parameters from all layers in a model.

    Args:
        fxp_model: Model to get its trainable parameters.
        fw_info: Framework information needed for keras kernel ops list.
        add_bias: Whether to include biases of the model (if there are) or not.

    Returns:
        A list of trainable variables in a model.
    """

    trainable_weights = []
    for layer in fxp_model.layers:
        if isinstance(layer, QuantizeWrapper) and isinstance(
                layer.quantize_config, (ActivationQuantizeConfigKD,
                                        WeightQuantizeConfigKD)):
            trainable_weights.extend(layer.quantize_config.get_trainable_quantizer_parameters())
            if add_bias:
                use_bias = layer.layer.get_config().get(USE_BIAS)
                if use_bias is not None and use_bias:
                    trainable_weights.append(layer.layer.bias)
            if isinstance(layer.layer, tuple(fw_info.kernel_ops)):
                for t in layer.trainable_weights:
                    if len(t.shape) > 1:
                        trainable_weights.append(t)
    return trainable_weights

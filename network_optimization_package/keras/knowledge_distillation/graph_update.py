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


import copy
from typing import Tuple, Any

import numpy as np
import tensorflow as tf
from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper

from network_optimization_package import common
from network_optimization_package.keras.constants import BIAS, USE_BIAS
from network_optimization_package.keras.quantizer.configs import ActivationQuantizeConfigKD, WeightQuantizeConfigKD
from tensorflow.keras.models import Model
from network_optimization_package.common.graph.base_graph import Graph


def update_graph_after_kd(fxp_model: Model,
                          graph: Graph,
                          power_of_two_constraint_flag: bool = True,
                          add_bias: bool = False) -> Graph:
    """
    After training using KD, update nodes in a graph with thresholds and weights
    of their corresponding layers in a model (weights are updated for Conv2D and DepthwiseConv2D nodes only).

    Args:
        fxp_model: Model to gets updates from.
        graph: Graph to update its nodes.
        power_of_two_constraint_flag: Whether updated thresholds should be constrained or not.
        add_bias: Whether to update the nodes biases or not.

    Returns:
        An updated graph.
    """

    graph = copy.copy(graph)

    for layer in fxp_model.layers:
        if isinstance(layer, QuantizeWrapper) and isinstance(
                layer.quantize_config, (ActivationQuantizeConfigKD,
                                        WeightQuantizeConfigKD)):
            node = graph.find_node_by_name(layer.layer.name)
            if len(node) != 1:
                common.Logger.error(f"Can't update KD graph due to missing layer named: {layer.layer.name}")
            node = node[0]
            weights, quant_config = layer.quantize_config.update_layer_quantization_params(layer)
            for weight_attr, weight in weights.items():
                node.set_weights_by_keys(weight_attr, weight.numpy())
            for config_attr, config_value in quant_config.items():
                if isinstance(layer.quantize_config, ActivationQuantizeConfigKD):
                    node.activation_quantization_cfg.set_quant_config_attr(config_attr, config_value)
                if isinstance(layer.quantize_config, WeightQuantizeConfigKD):
                    node.weights_quantization_cfg.set_quant_config_attr(config_attr, config_value)
            if add_bias:
                use_bias = layer.layer.get_config().get(USE_BIAS)
                if use_bias is not None and use_bias:
                    new_bias = layer.layer.bias.numpy()
                    node.set_weights_by_keys(BIAS, new_bias)

    return graph

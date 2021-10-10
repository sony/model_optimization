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


import copy
from typing import Tuple, Any

import numpy as np
import tensorflow as tf
from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper

from model_compression_toolkit import common
from model_compression_toolkit.keras.constants import BIAS, USE_BIAS
from model_compression_toolkit.keras.quantizer.configs import ActivationQuantizeConfigKD, WeightQuantizeConfigKD
from tensorflow.keras.models import Model
from model_compression_toolkit.common.graph.base_graph import Graph


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

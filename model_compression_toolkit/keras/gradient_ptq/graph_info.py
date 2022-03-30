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


import tensorflow as tf
from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper
from typing import Tuple, List

from model_compression_toolkit.common.graph.base_graph import Graph
from model_compression_toolkit.common.graph.base_node import BaseNode
from model_compression_toolkit.keras.constants import USE_BIAS
from model_compression_toolkit.keras.quantizer.gradient_ptq import WeightQuantizeConfig
from model_compression_toolkit.common.framework_info import FrameworkInfo
from tensorflow.keras.models import Model


def get_compare_points(input_graph: Graph) -> Tuple[List[BaseNode], List[str], List, List]:
    """
    Create a list of nodes with weights in a graph and a corresponding list
    of their names for tensors comparison purposes. Also outputs 2 list of activations
    prior information collected from batch normalization nodes (if exists)
    Args:
        input_graph: Graph to get its points to compare.

    Returns:
        A list of nodes in a graph
        A list of the their names.
        A list of nodes mean collected from BatchNorms in the graph
        A list of nodes std collected from BatchNorms in the graph
    """
    compare_points = []
    compare_points_mean = []
    compare_points_std = []
    compare_points_name = []
    for n in input_graph.get_topo_sorted_nodes():
        if len(n.weights) > 0:
            compare_points.append(n)
            compare_points_name.append(n.name)
            compare_points_std.append(n.prior_info.std_output)
            compare_points_mean.append(n.prior_info.mean_output)
    return compare_points, compare_points_name, compare_points_mean, compare_points_std


def get_trainable_parameters(fxp_model: Model,
                             fw_info: FrameworkInfo,
                             add_bias: bool = False) -> List[List[tf.Variable]]:
    """
    Get trainable parameters from all layers in a model

    Args:
        fxp_model: Model to get its trainable parameters.
        fw_info: Framework information needed for keras kernel ops list.
        add_bias: Whether to include biases of the model (if there are) or not.

    Returns:
        A list of trainable variables in a model. Each item is a list of a layers weights.
    """

    trainable_weights = []
    for layer in fxp_model.layers:
        if isinstance(layer, QuantizeWrapper) and isinstance(
                layer.quantize_config, WeightQuantizeConfig):
            # collect trainable weights per layer
            layer_trainable_weights = layer.quantize_config.get_trainable_quantizer_parameters()
            if add_bias:
                use_bias = isinstance(layer.layer, tuple(fw_info.kernel_ops)) and layer.layer.get_config().get(USE_BIAS)
                if use_bias is not None and use_bias:
                    layer_trainable_weights.append(layer.layer.bias)
            trainable_weights.append(layer_trainable_weights)

    return trainable_weights


def get_weights_for_loss(fxp_model: Model) -> Tuple[List[list], List[list]]:
    """
    Get all float and quantized kernels for the GPTQ loss

    Args:
        fxp_model: Model to get its float and quantized weights.

    Returns:
        A list of float kernels, each item is the float kernel of the layer
        A list of quantized kernels, each item is the quantized kernel of the layer
    """

    flp_weights_list = []
    fxp_weights_list = []
    for layer in fxp_model.layers:
        if isinstance(layer, QuantizeWrapper) and isinstance(
                layer.quantize_config, WeightQuantizeConfig):

            # collect pairs of float and quantized weights per layer
            _layer_flp_weights, _layer_fxp_weights = [], []
            for weight, quantizer, quantizer_vars in layer._weight_vars:
                _layer_flp_weights.append(weight)
                _layer_fxp_weights.append(quantizer(weight, training=False, weights=quantizer_vars))
            flp_weights_list.append(_layer_flp_weights)
            fxp_weights_list.append(_layer_fxp_weights)

    return flp_weights_list, fxp_weights_list

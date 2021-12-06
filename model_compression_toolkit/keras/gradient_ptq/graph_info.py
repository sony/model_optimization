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


import numpy as np
import tensorflow as tf
from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper
from typing import Tuple, List

from model_compression_toolkit.common.graph.base_graph import Graph
from model_compression_toolkit.common.graph.base_node import BaseNode
from model_compression_toolkit.keras.constants import USE_BIAS
from model_compression_toolkit.keras.quantizer.gradient_ptq import ActivationQuantizeConfig, WeightQuantizeConfig, ActivationAndWeightQuantizeConfig
from model_compression_toolkit.common.framework_info import FrameworkInfo
from tensorflow.keras.models import Model


def get_compare_points(input_graph: Graph) -> Tuple[List[BaseNode], List[str]]:
    """
    Create a list of nodes with weights in a graph and a corresponding list
    of their names for tensors comparison purposes.
    Args:
        input_graph: Graph to get its points to compare.

    Returns:
        A list of nodes in a graph, and a list of the their names.
    """
    compare_points = []
    compare_points_name = []
    for n in input_graph.get_topo_sorted_nodes():
        if len(n.weights) > 0:
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
                layer.quantize_config, (ActivationQuantizeConfig, ActivationAndWeightQuantizeConfig,
                                        WeightQuantizeConfig)):
            trainable_weights.extend(layer.quantize_config.get_trainable_quantizer_parameters())
            if add_bias:
                use_bias = isinstance(layer.layer, tuple(fw_info.kernel_ops)) and layer.layer.get_config().get(USE_BIAS)
                if use_bias is not None and use_bias:
                    trainable_weights.append(layer.layer.bias)
    return trainable_weights

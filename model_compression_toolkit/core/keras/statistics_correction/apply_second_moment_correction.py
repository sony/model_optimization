# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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
from typing import Any, Callable

from tensorflow.keras.layers import BatchNormalization
from tqdm import tqdm

import model_compression_toolkit.core.keras.constants as keras_constants
from model_compression_toolkit.core import CoreConfig
from model_compression_toolkit.core import common


def keras_apply_second_moment_correction(quantized_model: Any,
                                         core_config: CoreConfig,
                                         representative_data_gen: Callable,
                                         graph: common.Graph):
    """
    Apply second moment statistics correction to graph.

    Args:
        quantized_model: Framework's model to apply second moment correction on.
        core_config: QuantizationConfig of how the model should be quantized.
        representative_data_gen: Dataset to use for retrieving images for the models inputs.
        graph: Graph to update the parameters after the second moment correction.

    Returns:
        A function that applies second moment correction.
    """

    # Move every BN to train mode
    for layer in quantized_model.layers:
        if len(graph.find_node_by_name(layer.name)) > 0:
            node = graph.find_node_by_name(layer.name)[0]
            if isinstance(layer, BatchNormalization) and node.final_weights_quantization_cfg\
                    .weights_second_moment_correction:
                layer.trainable = True

    for data in tqdm(representative_data_gen()):
        quantized_model(data, training=True)

    # Move every BN to eval mode and update the corresponding BN node params in the graph
    for layer in quantized_model.layers:
        if len(graph.find_node_by_name(layer.name)) > 0:
            node = graph.find_node_by_name(layer.name)[0]
            if isinstance(layer, BatchNormalization) and node.final_weights_quantization_cfg\
                    .weights_second_moment_correction:
                layer.trainable = False
                bn_node_weights = {keras_constants.GAMMA: layer.gamma,
                                   keras_constants.BETA: layer.beta,
                                   keras_constants.MOVING_MEAN: layer.moving_mean,
                                   keras_constants.MOVING_VARIANCE: layer.moving_variance}
                node.weights = copy.deepcopy(bn_node_weights)
    return graph

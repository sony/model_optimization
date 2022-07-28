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

from typing import Any, Tuple

import tensorflow as tf
from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper

from model_compression_toolkit.core import common
from model_compression_toolkit.core.common import Logger
from model_compression_toolkit.core.common.constants import TENSORFLOW
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common.model_builder_mode import ModelBuilderMode
from model_compression_toolkit.core.keras.back2framework.model_builder import get_node_name_from_layer, \
    is_layer_fake_quant
from model_compression_toolkit.qat.keras.quantizer.config_factory import quantization_config_builder

from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.core.keras.constants import DEFAULT_TP_MODEL
from model_compression_toolkit.core.keras.back2framework.model_builder import model_builder as core_model_builder

from model_compression_toolkit import get_target_platform_capabilities

DEFAULT_KERAS_TPC = get_target_platform_capabilities(TENSORFLOW, DEFAULT_TP_MODEL)


def model_builder(graph: common.Graph,
                  fw_info: FrameworkInfo = DEFAULT_KERAS_INFO) -> Tuple[tf.keras.models.Model, Any]:
    #################################################
    # Prepare model for Quantization Aware Training
    #################################################
    quantized_model, user_info = core_model_builder(graph,
                                                    mode=ModelBuilderMode.QUANTIZED,
                                                    fw_info=fw_info)

    def _quantize(layer):
        nodes = graph.find_node_by_name(get_node_name_from_layer(layer))
        if len(nodes) == 1:
            node = nodes[0]
            return QuantizeWrapper(layer, quantization_config_builder(node, fw_info))
        elif is_layer_fake_quant(layer):
            return layer
        else:
            Logger.error(f"Mismatch between keras model and graph can't find node named: {get_node_name_from_layer(layer)}")

    # clone each layer in the model and apply _quantize to the layer.
    qat_model = tf.keras.models.clone_model(quantized_model, input_tensors=None, clone_function=_quantize)

    return qat_model, user_info
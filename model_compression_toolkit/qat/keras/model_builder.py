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
import copy

import tensorflow as tf
from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper

from model_compression_toolkit.core import common
from model_compression_toolkit.core.common import Logger
from model_compression_toolkit.core.common.constants import TENSORFLOW
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.keras.keras_implementation import KerasImplementation
from model_compression_toolkit.core.common.model_builder_mode import ModelBuilderMode
from model_compression_toolkit.core.keras.back2framework.model_builder import get_node_name_from_layer, \
    is_layer_fake_quant
from model_compression_toolkit.qat.keras.quantizer.config_factory import quantization_config_builder

from model_compression_toolkit.core.keras.constants import DEFAULT_TP_MODEL
from model_compression_toolkit.core.keras.back2framework.model_builder import model_builder as core_model_builder
from model_compression_toolkit.core.common.quantization.quantize_graph_weights import quantize_graph_weights

from model_compression_toolkit import get_target_platform_capabilities

DEFAULT_KERAS_TPC = get_target_platform_capabilities(TENSORFLOW, DEFAULT_TP_MODEL)


def is_qat_applicable(node: common.BaseNode, fw_info: FrameworkInfo) -> bool:
    """
    A function for deciding if a layer should be fine-tuned during QAT
    Args:
        node (BaseNode): Node for quantization decision
        fw_info (FrameworkInfo): Keras quantization information

    Returns:
        A boolean whether the layer is to be wrapped with a QuantizeWrapper
    """

    return fw_info.is_kernel_op(node.type) and node.is_weights_quantization_enabled()


def model_builder(graph: common.Graph,
                  fw_info: FrameworkInfo,
                  fw_impl: KerasImplementation) -> Tuple[tf.keras.models.Model, Any]:
    """
    Prepare model for Quantization Aware Training. Build a keras model and then wrap
    required layers with a QuantizeWrapper
    Args:
        graph (Graph): network graph to build
        fw_info (FrameworkInfo): Keras quantization information
        fw_impl: FrameworkImplementation object with a methods for keras implementation.

    Returns:
        A boolean whether the layer is to be wrapped with a QuantizeWrapper
    """

    #################################################
    # Prepare model for Quantization Aware Training
    #################################################

    # Quantize graph weights that are not to be fine-tuned during QAT
    graph_to_quantize = copy.deepcopy(graph)
    for n in graph_to_quantize.nodes:
        if is_qat_applicable(n, fw_info):
            n.final_weights_quantization_cfg.enable_weights_quantization = False
    quantized_tg = quantize_graph_weights(graph_to_quantize,
                                          fw_info=fw_info,
                                          fw_impl=fw_impl)

    # build keras model
    quantized_model, user_info = core_model_builder(quantized_tg,
                                                    mode=ModelBuilderMode.QUANTIZED,
                                                    fw_info=fw_info)

    # Wrap layers to be fine-tuned during QAT with QuantizeWrapper
    def _quantize(layer):
        nodes = graph.find_node_by_name(get_node_name_from_layer(layer))
        if len(nodes) == 1:
            node = nodes[0]
            if is_qat_applicable(node, fw_info):
                return QuantizeWrapper(layer, quantization_config_builder(node, fw_info))
            else:
                return layer
        elif is_layer_fake_quant(layer):
            # A fake quant layer was added in the core_model_builder to quantize the activations
            return layer
        else:
            Logger.error(f"Mismatch between keras model and graph can't find node named: {get_node_name_from_layer(layer)}")

    # clone each layer in the model and apply _quantize to the layer.
    qat_model = tf.keras.models.clone_model(quantized_model, input_tensors=None, clone_function=_quantize)

    return qat_model, user_info
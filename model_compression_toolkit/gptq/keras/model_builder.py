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


import tensorflow as tf

from model_compression_toolkit.core.common.model_builder_mode import ModelBuilderMode
from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper
from typing import Tuple, Any
from model_compression_toolkit.core.common.logger import Logger
from model_compression_toolkit.core import common
from model_compression_toolkit.gptq.common.gptq_config import GradientPTQConfig
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.gptq.keras.quantizer.config_factory import quantization_config_builder_gptq
from model_compression_toolkit.core.keras.back2framework.model_builder import get_node_name_from_layer, \
    is_layer_fake_quant
from model_compression_toolkit.core.keras.back2framework.model_builder import model_builder as core_model_builder


def model_builder(graph: common.Graph,
                  gptq_config: GradientPTQConfig,
                  mode: ModelBuilderMode = ModelBuilderMode.QUANTIZED,
                  append2output=None,
                  fw_info: FrameworkInfo = DEFAULT_KERAS_INFO) -> Tuple[tf.keras.models.Model, Any]:
    """
    Build a Keras model for GPTQ from a graph representing the model.
    The model is built by converting the graph nodes to Keras layers and applying them sequentially to get the model
    output tensors. The output tensors list and an input tensors list, then use to build the model.
    After the model is built in Keras, it is cloned to add quantization wrappers for GPTQ fine-tuning

    Args:
        graph: Graph to build its corresponding Keras model.
        gptq_config: GPTQ Configuration class.
        mode: Building mode. Read ModelBuilderMode description for more info.
        append2output: List of nodes or OutTensor objects. In float building mode,
        when the list contains nodes, all output tensors of all nodes are set as the model outputs.
        fw_info: Framework information (e.g., mapping from layers to their attributes to quantize).
        This is for passing the kernel attributes to the QuanteWrappers.

    Returns:
        A tuple of the model and an UserInformation object.
    """
    if gptq_config is None:
        Logger.exception("Building a model in GPTQ requires a GPTQ configuration as input")

    model, graph_user_info = core_model_builder(graph, mode, append2output, fw_info, return_float_outputs=True)

    def _quantize(layer):
        nodes = graph.find_node_by_name(get_node_name_from_layer(layer))
        if len(nodes) == 1:
            node = nodes[0]
            return QuantizeWrapper(layer, quantization_config_builder_gptq(node, fw_info, gptq_config))
        elif is_layer_fake_quant(layer):
            return layer
        else:
            raise Exception(
                f"Mismatch between keras model and graph can't find node named: {get_node_name_from_layer(layer)}")

    # clone each layer in the model and apply _quantize to the layer.
    model = tf.keras.models.clone_model(model, input_tensors=None, clone_function=_quantize)

    return model, graph_user_info

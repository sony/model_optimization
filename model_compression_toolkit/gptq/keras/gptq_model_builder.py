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

from typing import List, Tuple

import tensorflow as tf
from keras.models import Model

from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.common.user_info import UserInformation
from model_compression_toolkit.core.keras.back2framework.keras_model_builder import KerasModelBuilder, \
    is_layer_fake_quant, get_node_name_from_layer


from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper
from model_compression_toolkit.core import common
from model_compression_toolkit.gptq.common.gptq_config import GradientPTQConfig
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.gptq.keras.quantizer.config_factory import quantization_config_builder_gptq
from tensorflow.python.util.object_identity import Reference as TFReference


class GPTQKerasModelBuilder(KerasModelBuilder):
    """
    Builder of GPTQ Keras models.
    """

    def __init__(self,
                 graph: common.Graph,
                 gptq_config: GradientPTQConfig,
                 append2output=None,
                 fw_info: FrameworkInfo = DEFAULT_KERAS_INFO,
                 return_float_outputs: bool = True):
        """

        Args:
            graph: Graph to build the model from.
            gptq_config: Configuration for GPTQ optimization.
            append2output: Nodes to append to model's output.
            fw_info: Information about the specific framework of the model that is built.
            return_float_outputs: Whether the model returns float tensors or not.
        """

        super().__init__(graph,
                         append2output,
                         fw_info,
                         return_float_outputs)
        self.gptq_config = gptq_config

    def _quantize_node_activations(self,
                                   node: BaseNode,
                                   input_tensors: List[TFReference]) -> List[TFReference]:
        """
        Quantize node's activation given input tensors.

        Args:
            node: Node to quantize its outputs.
            input_tensors: Input tensors of the node.

        Returns:
            Output of the node.

        """

        return node.final_activation_quantization_cfg.quantize_node_output(input_tensors)

    def build_model(self) -> Tuple[Model, UserInformation]:
        """
        Build a Keras GPTQ model and return it.
        Returns: GPTQ Keras model.

        """
        model, user_info = super().build_model()

        def _quantize(layer):
            nodes = self.graph.find_node_by_name(get_node_name_from_layer(layer))
            if len(nodes) == 1:
                node = nodes[0]
                return QuantizeWrapper(layer, quantization_config_builder_gptq(node, self.fw_info, self.gptq_config))
            elif is_layer_fake_quant(layer):
                return layer
            else:
                raise Exception(
                    f"Mismatch between keras model and graph can't find node named: "
                    f"{get_node_name_from_layer(layer)}")

        # clone each layer in the model and apply _quantize to the layer.
        model = tf.keras.models.clone_model(model, input_tensors=None, clone_function=_quantize)

        return model, user_info

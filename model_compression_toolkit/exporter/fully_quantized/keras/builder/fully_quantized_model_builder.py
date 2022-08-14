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

from model_compression_toolkit.core.keras.quantizer.input_layer_quantize_transform import InputLayerWrapperTransform

from typing import List, Tuple

import tensorflow as tf
from keras.models import Model
from tensorflow.python.util.object_identity import Reference as TFReference
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit.default_8bit_quantize_configs import \
    NoOpQuantizeConfig
from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper

from model_compression_toolkit.core import common
from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common.user_info import UserInformation
from model_compression_toolkit.core.keras.back2framework.keras_model_builder import KerasModelBuilder, \
    is_layer_fake_quant, get_node_name_from_layer
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.exporter.fully_quantized.keras.builder.quantize_config_to_node import \
    get_quantization_config
from model_compression_toolkit.exporter.fully_quantized.keras.quantize_configs.activation_quantize_config import \
    ActivationQuantizeConfig
from model_compression_toolkit.exporter.fully_quantized.keras.quantize_configs.weights_activation_quantize_config import \
    WeightsActivationQuantizeConfig
from model_compression_toolkit.exporter.fully_quantized.keras.quantize_configs.weights_quantize_config import WeightsQuantizeConfig

from model_compression_toolkit.exporter.fully_quantized.keras.quantizers.fq_quantizer import FakeQuantQuantizer
from model_compression_toolkit.exporter.fully_quantized.keras.quantizers.uniform_quantizer import UniformQuantizer, \
    WeightsUniformQuantizer
import tensorflow_model_optimization.quantization.keras.graph_transformations.model_transformer as mt


class FullyQuantizedKerasModelBuilder(KerasModelBuilder):
    """
    Builder of fully-quantized Keras models.
    """

    def __init__(self,
                 graph: common.Graph,
                 append2output=None,
                 fw_info: FrameworkInfo = DEFAULT_KERAS_INFO,
                 return_float_outputs: bool = False):
        """

        Args:
            graph: Graph to build the model from.
            append2output: Nodes to append to model's output.
            fw_info: Information about the specific framework of the model that is built.
            return_float_outputs: Whether the model returns float tensors or not.
        """

        super().__init__(graph,
                         append2output,
                         fw_info,
                         return_float_outputs)

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
        return input_tensors

    def build_model(self) -> Tuple[Model, UserInformation]:
        """
        Build a Keras mixed-precision model and return it.
        Returns: Mixed-precision Keras model.

        """
        model, user_info = super().build_model()

        def _wrap_layer_with_quantize_config(layer):

            nodes = self.graph.find_node_by_name(get_node_name_from_layer(layer))

            if len(nodes) == 1:
                node = nodes[0]
                return QuantizeWrapper(layer, get_quantization_config(node,
                                                                      self.fw_info))

            elif is_layer_fake_quant(layer):
                return layer

            else:
                raise Exception(
                    f'Mismatch between keras model and graph cant find node named: '
                    f'{get_node_name_from_layer(layer)}')

        # clone each layer in the model and apply _wrap_layer_with_quantize_config to the layer.
        model = tf.keras.models.clone_model(model,
                                            input_tensors=None,
                                            clone_function=_wrap_layer_with_quantize_config)

        # We use a model transformer to wrap the input layer with QuantizeWrapper.
        # A model transformer allows to modify a layer in an existing model, by applying the given list of
        # transformers on the model (in this case,
        # we only apply single transformer - InputLayerQuantizeTransform)
        model_inputs = self.graph.get_inputs()

        input_transformer = mt.ModelTransformer(model, [InputLayerWrapperTransform(inp,
                                                                                   self.fw_info,
                                                                                   get_quantization_config(inp,
                                                                                                           self.fw_info),
                                                                                   self.get_custom_objects()
                                                                                   )
                                                        for inp in model_inputs])
        model = input_transformer.transform()[0]

        user_info.custom_objects = self.get_custom_objects()

        return model, user_info

    def get_custom_objects(self):
        return {"QuantizeWrapper":QuantizeWrapper,
                "WeightsActivationQuantizeConfig": WeightsActivationQuantizeConfig,
                "ActivationQuantizeConfig": ActivationQuantizeConfig,
                "WeightsQuantizeConfig": WeightsQuantizeConfig,
                "WeightsUniformQuantizer": WeightsUniformQuantizer,
                "UniformQuantizer": UniformQuantizer,
                "NoOpQuantizeConfig": NoOpQuantizeConfig,
                "FakeQuantQuantizer": FakeQuantQuantizer}






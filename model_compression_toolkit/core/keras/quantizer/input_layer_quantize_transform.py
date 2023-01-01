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
from typing import Dict

import keras.layers
from keras.engine.input_layer import InputLayer
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit.default_8bit_transforms import \
    InputLayerQuantize
from tensorflow_model_optimization.python.core.quantization.keras.graph_transformations import transforms
from tensorflow_model_optimization.python.core.quantization.keras.quantize_config import QuantizeConfig


from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.common.constants import INPUT_BASE_NAME


class InputLayerWrapperTransform(InputLayerQuantize):
    """
    Allows to configure an input layer with QuantizeWrapper given a QuantizeConfig object to wrap it.
    """

    def __init__(self,
                 input_layer: BaseNode,
                 quantize_config: QuantizeConfig,
                 custom_objects: Dict[str, type],
                 wrapper_class: type):
        """
        Transform for replacing input layers with a wrapped version of them to
        quantize them.

        Args:
            input_layer: Node that represents an input layer to wrap.
            quantize_config: QuantizeConfig to define the quantization of the layer.
            custom_objects: Custom objects dictionary for deserializing the layer.
            wrapper_class: A wrapper to use for wrapping the layer (exporter uses FullyQuantizedQuantizeWrapper,
            mixed-precision uses QuantizeWrapper).
        """
        super(InputLayerWrapperTransform, self).__init__()

        self.input_layer = input_layer
        self.name = self.input_layer.name
        self.quantize_config = quantize_config
        self.custom_objects = lambda: custom_objects
        self.wrapper_class = wrapper_class

    def pattern(self):
        return transforms.LayerPattern('InputLayer', config={'name': f'{self.name}_{INPUT_BASE_NAME}'})

    def replacement(self, match_layer):
        layer_wrapper = self.wrapper_class(InputLayer(input_shape=self.input_layer.input_shape),
                                           self.quantize_config)

        quantized_layer_name = f"quant_{self.name}"
        cfg_dict = keras.layers.serialize(layer_wrapper)

        cfg_dict['name'] = quantized_layer_name
        cfg_dict['config']['name'] = quantized_layer_name

        layer_node = transforms.LayerNode(cfg_dict, input_layers=[match_layer])

        return layer_node

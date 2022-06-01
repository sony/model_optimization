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

import keras.layers
from keras.engine.input_layer import InputLayer
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit.default_8bit_transforms import \
    InputLayerQuantize
from tensorflow_model_optimization.python.core.quantization.keras.graph_transformations import transforms

from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper

from model_compression_toolkit.core.keras.quantizer.mixed_precision.quantization_config_factory import \
  quantization_config_builder_mixed_precision
from model_compression_toolkit.core.keras.quantizer.mixed_precision.selective_quantize_config import SelectiveQuantizeConfig


class InputLayerMixedPrecisionTransform(InputLayerQuantize):
    """
    Allows to configure an input layer with QuantizeWrapper for mixed-precision quantization.
    """

    def __init__(self, input_layer, fw_info):
        super(InputLayerMixedPrecisionTransform, self).__init__()

        self.input_layer = input_layer
        self.fw_info = fw_info
        self.name = self.input_layer.name

    def pattern(self):
        return transforms.LayerPattern('InputLayer', config={'name': self.name})

    def replacement(self, match_layer):
        layer_wrapper = QuantizeWrapper(InputLayer(input_shape=self.input_layer.input_shape),
                                        quantization_config_builder_mixed_precision(self.input_layer, self.fw_info))
        quantized_layer_name = f"quant_{self.name}"
        cfg_dict = keras.layers.serialize(layer_wrapper)

        cfg_dict['name'] = quantized_layer_name
        cfg_dict['config']['name'] = quantized_layer_name

        layer_node = transforms.LayerNode(cfg_dict, input_layers=[match_layer])

        return layer_node

    def custom_objects(self):
        return {
          'QuantizeWrapper': QuantizeWrapper,
          'SelectiveQuantizeConfig': SelectiveQuantizeConfig,
        }

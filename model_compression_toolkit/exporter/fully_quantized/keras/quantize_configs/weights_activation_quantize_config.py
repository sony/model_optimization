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

from typing import List, Tuple, Any, Dict

import tensorflow as tf
from tensorflow import Tensor
from tensorflow_model_optimization.python.core.quantization.keras.quantizers import Quantizer

# As from Tensorflow 2.6, keras is a separate package and some classes should be imported differently.
from model_compression_toolkit.exporter.fully_quantized.keras.quantize_configs.activation_quantize_config import \
    ActivationQuantizeConfig
from model_compression_toolkit.exporter.fully_quantized.keras.quantize_configs.weights_quantize_config import \
    WeightsQuantizeConfig

if tf.__version__ < "2.6":
    from tensorflow.python.keras.layers import Layer
else:
    from keras.engine.base_layer import Layer
from tensorflow.python.training.tracking.data_structures import ListWrapper
from tensorflow_model_optimization.python.core.quantization.keras.quantize_config import QuantizeConfig


class WeightsActivationQuantizeConfig(QuantizeConfig):
    """
    QuantizeConfig to quantize a layer's activations and weights.
    """

    def __init__(self,
                 activation_quantizer: Quantizer,
                 w_quantizer: Quantizer,
                 weight_attrs: List[str] = None):
        """

        Args:
            activation_quantizer: Quantizer for activations.
            w_quantizer: Quantizer for weights.
            weight_attrs: Weights attributes to quantize.
        """
        self.act_config = ActivationQuantizeConfig(activation_quantizer=activation_quantizer)
        self.weights_config = WeightsQuantizeConfig(w_quantizer=w_quantizer,
                                                    weight_attrs=weight_attrs)


    def get_config(self) -> Dict[str,Any]:
        """

        Returns: Configuration of WeightsActivationQuantizeConfig

        """
        return {"activation_quantizer": self.act_config.activation_quantizer,
                "w_quantizer": self.weights_config.w_quantizer,
                "weight_attrs": self.weights_config.weight_attrs}

    def get_weights_and_quantizers(self, layer: Layer) -> List[Tuple[Tensor, Any]]:
        """
        Get the layer's weights to quantize and quantizers.

        Args:
            layer: Layer wrapped with this WeightsQuantizeConfig

        Returns:
            List of weights and quantizers to quantize these weights.
        """
        return self.weights_config.get_weights_and_quantizers(layer)

    def get_activations_and_quantizers(self, layer: Layer) -> list:
        """
        Get the layer's activations to quantize and quantizers.

        Args:
            layer: Layer wrapped with this WeightsActivationQuantizeConfig

        Returns:
            List of activation tensors and quantizers to quantize them.
        """
        return self.act_config.get_activations_and_quantizers(layer)

    def set_quantize_weights(self, layer: Layer, quantize_weights: List[Tensor]):
        """
        Set layer's weights with quantized weights.

        Args:
            layer: Layer wrapped with this WeightsQuantizeConfig
            quantize_weights: Quantized weights to set to the layer

        Returns:
            None
        """
        self.weights_config.set_quantize_weights(layer, quantize_weights)

    def set_quantize_activations(self, layer, quantize_activations: ListWrapper):
        """
        Set layer's activations with quantized activations.

        Args:
            layer: Layer wrapped with this WeightsActivationQuantizeConfig
            quantize_activations: Quantized activation to set to the layer

        Returns:
            None
        """
        self.act_config.set_quantize_activations(layer, quantize_activations)

    def get_output_quantizers(self, layer: Layer) -> List[Quantizer]:
        """
        Quantize layer's outputs.

        Args:
            layer: Layer to quantize its activations.

        Returns: List of activation quantizers.

        """
        return self.act_config.get_output_quantizers(layer)

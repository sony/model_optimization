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
from packaging import version

# As from Tensorflow 2.6, keras is a separate package and some classes should be imported differently.
if version.parse(tf.__version__) < version.parse("2.6"):
    from tensorflow.python.keras.layers import Layer
else:
    from keras.engine.base_layer import Layer
from tensorflow.python.training.tracking.data_structures import ListWrapper
from tensorflow_model_optimization.python.core.quantization.keras.quantize_config import QuantizeConfig


class WeightsQuantizeConfig(QuantizeConfig):
    """
    QuantizeConfig to quantize a layer's weights.
    """

    def __init__(self,
                 w_quantizer: Quantizer,
                 weight_attrs: List[str] = None):
        """

        Args:
            w_quantizer: Quantizer for weights.
            weight_attrs: Weights attributes to quantize.
        """

        self.weight_attrs = weight_attrs
        self.w_quantizer = w_quantizer

    def get_config(self) -> Dict[str, Any]:
        """

        Returns: Configuration of WeightsQuantizeConfig

        """
        return {'w_quantizer': self.w_quantizer,
                'weight_attrs': self.weight_attrs}

    def get_weights_and_quantizers(self, layer: Layer) -> List[Tuple[Tensor, Any]]:
        """
        Get the layer's weights to quantize and quantizers.

        Args:
            layer: Layer wrapped with this WeightsQuantizeConfig

        Returns:
            List of weights and quantizers to quantize these weights.
        """
        return [(getattr(layer, self.weight_attrs[i]),
                 self.w_quantizer) for i in range(len(self.weight_attrs))]

    def get_activations_and_quantizers(self, layer: Layer) -> list:
        # For configurable activations we use get_output_quantizers,
        # Therefore, we do not need to implement this method.
        return []

    def set_quantize_weights(self, layer: Layer, quantize_weights: List[Tensor]):
        """
        Set layer's weights with quantized weights.

        Args:
            layer: Layer wrapped with this WeightsQuantizeConfig
            quantize_weights: Quantized weights to set to the layer

        Returns:
            None
        """
        if len(self.weight_attrs) != len(quantize_weights):
            raise ValueError(
                '`set_quantize_weights` called on layer {} with {} '
                'weight parameters, but layer expects {} values.'.format(
                    layer.name, len(quantize_weights), len(self.weight_attrs)))

        for weight_attr, weight in zip(self.weight_attrs, quantize_weights):
            current_weight = getattr(layer, weight_attr)
            if current_weight.shape != weight.shape:
                raise ValueError('Existing layer weight shape {} is incompatible with'
                                 'provided weight shape {}'.format(
                    current_weight.shape, weight.shape))

            setattr(layer, weight_attr, weight)

    def set_quantize_activations(self, layer, quantize_activations: ListWrapper):
        pass

    def get_output_quantizers(self, layer: Layer) -> List[Quantizer]:
        return []

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


class ActivationQuantizeConfig(QuantizeConfig):
    """
    QuantizeConfig to quantize a layer's activations.
    """

    def __init__(self,
                 activation_quantizer: Quantizer):
        """

        Args:
            activation_quantizer: Quantizer for quantization the layer's activations.
        """

        self.activation_quantizer = activation_quantizer


    def get_config(self) -> Dict[str, Any]:
        """

        Returns: Configuration of ActivationQuantizeConfig

        """
        return {
            'activation_quantizer': self.activation_quantizer}

    def get_weights_and_quantizers(self, layer: Layer) -> List[Tuple[Tensor, Any]]:
        return []

    def get_activations_and_quantizers(self, layer: Layer) -> list:
        # For configurable activations we use get_output_quantizers,
        # Therefore, we do not need to implement this method.
        return []

    def set_quantize_weights(self, layer: Layer, quantize_weights: List[Tensor]):
        pass

    def set_quantize_activations(self, layer, quantize_activations: ListWrapper):
        pass

    def get_output_quantizers(self, layer: Layer) -> List[Quantizer]:
        """
        Quantize layer's outputs.

        Args:
            layer: Layer to quantize its activations.

        Returns: List of activation quantizers.

        """
        return [self.activation_quantizer]

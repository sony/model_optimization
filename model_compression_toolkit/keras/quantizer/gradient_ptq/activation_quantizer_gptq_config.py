# Copyright 2021 Sony Semiconductors Israel, Inc. All rights reserved.
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

from typing import List, Any, Dict

from tensorflow.python.training.tracking.data_structures import ListWrapper

from model_compression_toolkit.common.constants import THRESHOLD
from model_compression_toolkit.keras.quantizer.gradient_ptq.activation_quantizer import TrainableQuantizer
from model_compression_toolkit.keras.quantizer.gradient_ptq.base_quantizer_gptq_config import BaseQuantizeConfig
import tensorflow as tf
# As from Tensorflow 2.6, keras is a separate package and some classes should be imported differently.
if tf.__version__ < "2.6":
    from tensorflow.python.keras.layers import Layer
else:
    from keras.engine.base_layer import Layer
from tensorflow import Tensor
from tensorflow_model_optimization.python.core.quantization.keras.quantizers import Quantizer


class ActivationQuantizeConfig(BaseQuantizeConfig):
    """
    QuantizeConfig to quantize the activations of a layer using a TrainableQuantizer.
    """

    def __init__(self,
                 activation_quantization_params: dict,
                 signed: bool,
                 num_bits: int = 8):
        """
        Initialize a TrainableQuantizer and set as the activation quantizer.

        Args:
            activation_quantization_params: Parameters to use for quantization.
            signed: Quantization range is signed or unsigned.
            num_bits: Number of bits to use for quantization.
        """
        threshold_values = activation_quantization_params.get(THRESHOLD)
        self.activation_quantizer = TrainableQuantizer(num_bits=num_bits,
                                                       per_axis=False,
                                                       threshold_values=threshold_values,
                                                       signed=signed,
                                                       trainable=False)

    def get_weights_and_quantizers(self, layer: Layer) -> list:
        return []

    def get_activations_and_quantizers(self, layer: Layer) -> list:
        return []

    def set_quantize_weights(self, layer: Layer, quantize_weights: List[Tensor]):
        pass  # pragma: no cover

    def set_quantize_activations(self, layer: Layer, quantize_activations: ListWrapper):
        pass  # pragma: no cover

    def get_output_quantizers(self, layer: Layer) -> List[Quantizer]:
        """
        Get the activation quantizer.
        Args:
            layer: The layer the ActivationQuantizeConfig wraps.

        Returns:
            The ActivationQuantizeConfig activation quantizer.
        """
        return [self.activation_quantizer]

    def update_layer_quantization_params(self, layer):
        """
        A Function to calculate the needed change in attributes in NodeQuantizationConfig after retraining.
        Usually a function of the config quantizers.

        Args:
            layer: layer being quantized.

        Returns:
            3 dictionaries describing the change in layer's weights, weights config, activation config
            that changed during GPTQ retraining.
            Keys must match NodeQuantizationConfig attributes

        """
        quant_config = {'activation_quantization_params': self.activation_quantizer.calc_quant_config(layer)}

        return {}, {}, quant_config

    def get_trainable_quantizer_parameters(self):
        """
        A function to get a list trainable of trainable parameters for GPTQ retraining from config quantizers

        Returns:
            A list of trainable Tensors

        """
        return self.activation_quantizer.get_trainable_parameters()

    def get_config(self) -> Dict[str, Any]:
        """Returns the config used to serialize `QuantizeConfig`."""
        return {}

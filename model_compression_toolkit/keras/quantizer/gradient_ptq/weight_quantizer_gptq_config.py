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

from typing import List, Tuple, Any, Dict

from tensorflow import Tensor
import tensorflow as tf

# As from Tensorflow 2.6, keras is a separate package and some classes should be imported differently.
if tf.__version__ < "2.6":
    from tensorflow.python.keras.layers import Layer
else:
    from keras.engine.base_layer import Layer


from tensorflow.python.training.tracking.data_structures import ListWrapper
from tensorflow_model_optimization.python.core.quantization.keras.quantizers import Quantizer

from model_compression_toolkit.keras.quantizer.gradient_ptq.weight_quantizer import TrainableWeightQuantizer
from model_compression_toolkit.keras.quantizer.gradient_ptq.base_quantizer_gptq_config import BaseQuantizeConfig
from model_compression_toolkit.keras.constants import KERNEL
import numpy as np


class WeightQuantizeConfig(BaseQuantizeConfig):
    """
    QuantizeConfig to quantize the weights of a layer using a TrainableQuantizer.
    """

    def __init__(self,
                 weight_attrs: List[str],
                 threshold_values: np.ndarray,
                 weight_channel_axis: int,
                 num_bits: int,
                 max_lsbs_change: int = 8):
        """
        Initialize a TrainableQuantizer and set as the weights quantizer.
        Args:
            weight_attrs: Attributes of the layer's weights to quantize.
            threshold_values: Thresholds to use for quantization.
            weight_channel_axis: Channel index to quantize when quantizing per-channel.
            num_bits: Number of bits to use for quantization.
        """

        self.weight_attrs = weight_attrs
        self.weight_quantizer = TrainableWeightQuantizer(num_bits=num_bits,
                                                         per_axis=len(threshold_values.flatten()) > 1,
                                                         threshold_values=threshold_values,
                                                         signed=True,
                                                         quantization_axis=weight_channel_axis,
                                                         max_lsbs_change=max_lsbs_change)

    def get_weights_and_quantizers(self, layer: Layer) -> List[Tuple[Tensor, Quantizer]]:
        """
        Get a list of tuples with weights and the weight quantizer.
        The layer's attributes are used to get the weights.
        Args:
            layer: The layer the WeightQuantizeConfig wraps.

        Returns:
            List of tuples of the layer's weights and the weight quantizer.
        """
        return [(getattr(layer, weight_attr), self.weight_quantizer)
                for weight_attr in self.weight_attrs]

    def get_activations_and_quantizers(self, layer: Layer) -> list:
        return []

    def set_quantize_weights(self, layer: Layer, quantize_weights: List[Tensor]):
        """
        Set the layer weights with new passed weights.
        Args:
            layer: Layer to set its attributes.
            quantize_weights: Quantized weights to set as new weights.

        """
        if len(self.weight_attrs) != len(quantize_weights):
            raise ValueError(
                '`set_quantize_weights` called on layer {} with {} '
                'weight parameters, but layer expects {} values.'.format(
                    layer.name, len(quantize_weights), len(self.weight_attrs)))  # pragma: no cover

        for weight_attr, weight in zip(self.weight_attrs, quantize_weights):
            current_weight = getattr(layer, weight_attr)
            if current_weight.shape != weight.shape:
                raise ValueError('Existing layer weight shape {} is incompatible with'
                                 'provided weight shape {}'.format(
                    current_weight.shape, weight.shape))  # pragma: no cover

            setattr(layer, weight_attr, weight)

    def set_quantize_activations(self, layer, quantize_activations: ListWrapper):
        pass

    def get_output_quantizers(self, layer: Layer) -> list:
        return []

    @classmethod
    def from_config(cls, config: dict):
        """
        Instantiates a `WeightQuantizeConfig` from its config.

        Args:
            config: Output of `get_config()`.

        Returns:
            A `WeightQuantizeConfig` instance.
        """

        return cls(**config)

    def get_config(self) -> Dict[str, Any]:
        """
        Returns: The WeightQuantizeConfig configuration.
        """
        return {
            'weight_attrs': self.weight_attrs,
        }

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
        weights = {}
        for weight_attr, (_weight, _quantizer) in zip(self.weight_attrs, self.get_weights_and_quantizers(layer.layer)):
            # TODO: elad: take weight_attr into configuration in case of more than 1 weight per layer.
            # TODO: elad: current weight_attr is actually weight_attr_framework. need to add weight_attr_hptq (KERNEL)
            weights.update({KERNEL: _weight})

        quant_config = {'weights_quantization_params': self.weight_quantizer.calc_quant_config(layer)}

        return weights, quant_config, {}

    def get_trainable_quantizer_parameters(self):
        """
        A function to get a list trainable of trainable parameters for GPTQ retraining from config quantizers

        Returns:
            A list of trainable Tensors

        """
        return self.weight_quantizer.get_trainable_parameters()

    def __eq__(self, other: Any) -> bool:
        """
        Check whether it equals to another object or not.
        """
        if not isinstance(other, WeightQuantizeConfig):
            return False

        return (self.weight_attrs == other.weight_attrs and
                self.weight_quantizer == other.weight_quantizer)

    def __ne__(self, other: Any) -> bool:
        """
        Check whether it differs from another object or not.
        """
        return not self.__eq__(other)

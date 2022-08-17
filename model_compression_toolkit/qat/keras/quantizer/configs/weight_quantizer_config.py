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

import numpy as np
from tensorflow import Tensor
import tensorflow as tf

from model_compression_toolkit.core.common.constants import THRESHOLD, RANGE_MIN, RANGE_MAX

# As from Tensorflow 2.6, keras is a separate package and some classes should be imported differently.


if tf.__version__ < "2.6":
    from tensorflow.python.keras.layers import Layer
else:
    from keras.engine.base_layer import Layer

from model_compression_toolkit.core.common.logger import Logger
from tensorflow.python.training.tracking.data_structures import ListWrapper
from tensorflow_model_optimization.python.core.quantization.keras.quantizers import Quantizer

from model_compression_toolkit.qat.keras.quantizer.configs.base_quantizer_config import BaseQuantizeConfig
from model_compression_toolkit.core.keras.constants import KERNEL

from model_compression_toolkit.qat.keras.quantizer.ste_rounding.symmetric_ste import STEWeightQuantizer
from model_compression_toolkit.qat.keras.quantizer.ste_rounding.uniform_ste import STEUniformWeightQuantizer
from model_compression_toolkit.core.common.target_platform.op_quantization_config import QuantizationMethod
from model_compression_toolkit.core import common
from model_compression_toolkit.qat.common import WEIGHTS_QUANTIZATION_PARAMS


class WeightQuantizeConfig(BaseQuantizeConfig):
    """
    QuantizeConfig to quantize the weights of a layer using a TrainableQuantizer.
    """

    def __init__(self,
                 weight_attrs: List[str],
                 num_bits,
                 channels_axis,
                 quantization_method: QuantizationMethod,
                 quantization_params: Dict):
        """
        Initialize a TrainableQuantizer and set as the weights quantizer.
        Args:
            weight_attrs: Attributes of the layer's weights to quantize.
            num_bits: number of bits to quantize
            channels_axis: axis of the channels in the tensor
            quantization_method (QuantizationMethod): quantization method, either SYMMETRIC or POWER_OF_TWO
            quantization_params: quantization params for the quantization method.
        """

        self.weight_attrs = weight_attrs
        self.num_bits = num_bits
        self.weight_channel_axis = channels_axis
        self.quantization_method = quantization_method
        self.quantization_params = quantization_params

        if quantization_method in [QuantizationMethod.SYMMETRIC,
                                   QuantizationMethod.POWER_OF_TWO]:
            is_power_of_two = QuantizationMethod.POWER_OF_TWO == quantization_method
            threshold_values = quantization_params.get(THRESHOLD)
            self.weight_quantizer = STEWeightQuantizer(num_bits=num_bits,
                                                       per_axis=len(
                                                           threshold_values.flatten()) > 1,
                                                       threshold_values=threshold_values,
                                                       signed=True,
                                                       power_of_two=is_power_of_two,
                                                       quantization_axis=self.weight_channel_axis)
        elif quantization_method in [QuantizationMethod.UNIFORM]:
            min_values = quantization_params.get(RANGE_MIN)
            max_values = quantization_params.get(RANGE_MAX)
            self.weight_quantizer = STEUniformWeightQuantizer(num_bits=num_bits,
                                                              per_axis=len(
                                                                  max_values.flatten()) > 1,
                                                              min_values=min_values,
                                                              max_values=max_values,
                                                              signed=True,
                                                              quantization_axis=self.weight_channel_axis)
        else:
            common.Logger.error(f'Weight quantization method not implemented: {quantization_method}')

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
            Logger.error(f"`set_quantize_weights` called on layer {layer.name} with {len(quantize_weights)} weight parameters, but layer expects {len(self.weight_attrs)} values.")  # pragma: no cover

        for weight_attr, weight in zip(self.weight_attrs, quantize_weights):
            current_weight = getattr(layer, weight_attr)
            if current_weight.shape != weight.shape:
                Logger.error(f"Existing layer weight shape {current_weight.shape} is incompatible with provided weight shape {weight.shape}")  # pragma: no cover

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

        config['quantization_method'] = QuantizationMethod(config['quantization_method'])
        if THRESHOLD in config['quantization_params']:
            config['quantization_params'][THRESHOLD] = np.array(config['quantization_params'][THRESHOLD])
        if RANGE_MIN in config['quantization_params']:
            config['quantization_params'][RANGE_MIN] = np.array(config['quantization_params'][RANGE_MIN])
        if RANGE_MAX in config['quantization_params']:
            config['quantization_params'][RANGE_MAX] = np.array(config['quantization_params'][RANGE_MAX])
        return cls(**config)

    def get_config(self) -> Dict[str, Any]:
        """
        Returns: The WeightQuantizeConfig configuration.
        """
        return {
            'weight_attrs': self.weight_attrs,
            'num_bits': self.num_bits,
            'channels_axis': self.weight_channel_axis,
            'quantization_method': self.quantization_method,
            'quantization_params': self.quantization_params,
        }

    def update_layer_quantization_params(self, layer: Layer) -> (Dict[str, tf.Tensor], Dict[str, Dict], Dict):
        """
        A Function to calculate the needed change in attributes in NodeQuantizationConfig after retraining.
        Usually a function of the config quantizers.

        Args:
            layer: layer being quantized.

        Returns:
            3 dictionaries describing the change in layer's weights, weights config, activation config
            that changed during QAT retraining.
            Keys must match NodeQuantizationConfig attributes

        """
        weights = {}
        for weight, quantizer, quantizer_vars in layer._weight_vars:
            weights.update({KERNEL: quantizer(weight, training=False, weights=quantizer_vars)})

        quant_config = {WEIGHTS_QUANTIZATION_PARAMS: self.weight_quantizer.get_quant_config(layer)}

        return weights, quant_config, {}

    def get_trainable_quantizer_parameters(self) -> List[tf.Tensor]:
        """
        A function to get a list trainable of trainable parameters for QAT retraining from config quantizers

        Returns:
            A list of trainable Tensors

        """
        return self.weight_quantizer.get_trainable_parameters()

    def get_quantization_variable(self) -> List[tf.Tensor]:
        """
        This function return a list of quantizer parameters.
        Returns: A list of the quantizer parameters
        """
        return self.weight_quantizer.get_quantization_variable()

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

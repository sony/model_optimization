# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
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

from typing import Dict, Any, List

import numpy as np
from tensorflow import Tensor
import tensorflow as tf
from packaging import version

from model_compression_toolkit.core.keras.mixed_precision.configurable_quant_id import ConfigurableQuantizerIdentifier
from model_compression_toolkit.logger import Logger

if version.parse(tf.__version__) < version.parse("2.6"):
    from tensorflow.python.keras.layers import Layer  # pragma: no cover
else:
    from keras.engine.base_layer import Layer
from mct_quantizers.keras.quantizers import BaseKerasInferableQuantizer

from model_compression_toolkit.target_platform_capabilities.target_platform import QuantizationMethod
from mct_quantizers import QuantizationTarget
from mct_quantizers import mark_quantizer


# TODO: set TF_FOUND and imports accordingly



@mark_quantizer(quantization_target=QuantizationTarget.Weights,
                quantization_method=[QuantizationMethod.POWER_OF_TWO, QuantizationMethod.SYMMETRIC,
                                     QuantizationMethod.UNIFORM, QuantizationMethod.LUT_POT_QUANTIZER,
                                     QuantizationMethod.LUT_SYM_QUANTIZER],
                quantizer_type=ConfigurableQuantizerIdentifier.CONFIGURABLE_ID)
class ConfigurableWeightsQuantizer(BaseKerasInferableQuantizer):
    """
    TODO: update documentation
    Trainable symmetric quantizer to quantize a layer weights.
    """

    def __init__(self,
                 node_q_cfg,
                 float_weights,
                 weight_attrs,
                 max_candidate_idx):
        """
        Initialize a STEWeightGPTQQuantizer object with parameters to use for the quantization.

        Args:
            quantization_config: Trainable weights quantizer config.
        """

        super(ConfigurableWeightsQuantizer, self).__init__()

        # self.node_q_cfg = quantization_config.node_q_cfg
        # self.float_weights = quantization_config.float_weights
        # self.weight_attrs = quantization_config.weight_attrs
        # self.max_candidate_idx = quantization_config.max_candidate_idx

        self.node_q_cfg = node_q_cfg
        self.float_weights = float_weights
        self.weight_attrs = weight_attrs
        self.max_candidate_idx = max_candidate_idx

        # Make sure the candidates configurations arrived in descending order.
        curmax = (np.inf, np.inf)
        n_candidate_bits = [(x.weights_quantization_cfg.weights_n_bits, x.activation_quantization_cfg.activation_n_bits)
                            for x in self.node_q_cfg]
        for candidate_bits in n_candidate_bits:
            assert candidate_bits < curmax
            curmax = candidate_bits

        # TODO: do we need to care about activation quantization here? (can be removed IMO)
        for qc in self.node_q_cfg:
            assert qc.weights_quantization_cfg.enable_weights_quantization == \
                   self.node_q_cfg[0].weights_quantization_cfg.enable_weights_quantization \
                   and qc.activation_quantization_cfg.enable_activation_quantization == \
                   self.node_q_cfg[0].activation_quantization_cfg.enable_activation_quantization, \
                "Candidates with different weights/activation enabled properties is currently not supported"

        # self.enable_weights_quantization = self.node_q_cfg[0].weights_quantization_cfg.enable_weights_quantization
        # self.enable_activation_quantization = self.node_q_cfg[
        #     0].activation_quantization_cfg.enable_activation_quantization

        # Initialize quantized weights for each weight that should be quantized.
        self.quantized_weights = []
        # if self.enable_weights_quantization:
        # for float_weight in self.float_weights:
        for qc in self.node_q_cfg:
            qc_weights = qc.weights_quantization_cfg
            q_weight = qc_weights.weights_quantization_fn(self.float_weights,
                                                          qc_weights.weights_n_bits,
                                                          True,
                                                          qc_weights.weights_quantization_params,
                                                          qc_weights.weights_per_channel_threshold,
                                                          qc_weights.weights_channels_axis)

            self.quantized_weights.append(tf.Variable(q_weight,
                                                      trainable=False,
                                                      dtype=tf.float32))

        self.active_quantization_config_index = self.max_candidate_idx


    def set_weights_bit_width_index(self,
                                    index: int,
                                    attr: str = None):
        """
        Change the "active" bitwidth index the SelectiveQuantizeConfig uses, so
        a different quantized weight will be used.
        If attr is passed, only the quantizer that was created for this attribute will be configured.
        Otherwise, all quantizers the SelectiveQuantizeConfig holds will be configured
        using the passed index.

        Args:
            index: Bitwidth index to use.
            attr: Name of the layer's attribute to configure its corresponding quantizer.

        """

        # if self.enable_weights_quantization:
        # if attr is None:  # set bit width to all weights of the layer
        #     for q in self.weight_quantizers:
        #         q._set_active_quantization_config_index(index)
        # else:  # set bit width to a specific attribute
        #     i = self.weight_attrs.index(attr)
        #     q = self.weight_quantizers[i]
        #     q._set_active_quantization_config_index(index)
        assert index < len(
            self.node_q_cfg), f'Quantizer has {len(self.node_q_cfg)} ' \
                              f'possible nbits. Can not set ' \
                              f'index {index}'
        self.active_quantization_config_index = index

    def _set_active_quantization_config_index(self, index: int):
        """
        Set an index to use for the quantized weight the quantizer returns
        when requested.

        Args:
            index: Index of a candidate quantization configuration to use its quantized
            version of the float weight.

        """
        assert index < len(
            self.node_q_cfg), f'Quantizer has {len(self.node_q_cfg)} ' \
                                      f'possible nbits. Can not set ' \
                                      f'index {index}'
        self.active_quantization_config_index = index

    def set_quantize_weights(self, layer: Layer, quantize_weights: List[Tensor]):
        """
        Set the layer weights with new passed weights.
        Args:
            layer: Layer to set its attributes.
            quantize_weights: Quantized weights to set as new weights.

        """
        # if self.enable_weights_quantization:
        if len(self.weight_attrs) != len(quantize_weights):
            Logger.error('`set_quantize_weights` called on layer {} with {} '  # pragma: no cover
                         'weight parameters, but layer expects {} values.'.format(layer.name, len(quantize_weights),
                                                                                  len(self.weight_attrs)))

        for weight_attr, weight in zip(self.weight_attrs, quantize_weights):
            current_weight = getattr(layer, weight_attr)
            if current_weight.shape != weight.shape:
                Logger.error('Existing layer weight shape {} is incompatible with'  # pragma: no cover
                             'provided weight shape {}'.format(current_weight.shape, weight.shape))

            setattr(layer, weight_attr, weight)

    def __call__(self,
                 inputs: tf.Tensor,
                 training: bool):
        """
        Method to return the quantized weight. This method is called
        when the framework needs to quantize a float weight, and is expected to return the quantized
        weight. Since we already quantized the weight in all possible bitwidths, we do not
        quantize it again, and simply return the quantized weight according to the current
        active_quantization_config_index.

        Returns:
            Quantized weight, that was quantized using number of bits that is in a
            specific quantization configuration candidate (the candidate's index is the
            index that is in active_quantization_config_index the quantizer holds).
        """

        return self.quantized_weights[self.active_quantization_config_index]

    def get_config(self) -> Dict[str, Any]:
        """
        Returns: The SelectiveQuantizeConfig configuration.
        """
        # TODO: check if something is missing and verify signature against parent class
        return {
            'weight_attrs': self.weight_attrs,
            'float_weights': self.float_weights,
            'node_q_cfg': self.node_q_cfg
        }

        # def get_config(self) -> Dict[str, Any]:  # pragma: no cover
        #     """
        #     Returns: Configuration of TrainableQuantizer.
        #     """
        #
        #     return {
        #         'node_q_cfg': self.node_q_cfg,
        #         'float_weight': self.float_weight,
        #         'quantizer_fn_list': self.quantizer_fn_list,
        #         'quantized_weights': self.quantized_weights,
        #         'active_quantization_config_index': self.active_quantization_config_index
        #     }

    # TODO: Do we need to implement equal?
    # def __eq__(self, other: Any) -> bool:  # pragma: no cover
    #     """
    #     Check if equals to another object.
    #     Args:
    #         other: Other object to compare.
    #
    #     Returns:
    #         Whether they are equal or not.
    #     """
    #     if not isinstance(other, SelectiveWeightsQuantizer):
    #         return False
    #
    #     return (self.node_q_cfg == other.node_q_cfg and
    #             self.float_weight == other.float_weight and
    #             self.quantizer_fn_list == other.quantizer_fn_list and
    #             self.self.quantized_weights == other.self.quantized_weights and
    #             self.active_quantization_config_index == other.active_quantization_config_index)
    #
    # def __ne__(self, other: Any) -> bool:  # pragma: no cover
    #     """
    #     Check if not equals to another object.
    #     Args:
    #         other: Other object to compare.
    #
    #     Returns:
    #         Whether they are differ or not.
    #     """
    #     return not self.__eq__(other)
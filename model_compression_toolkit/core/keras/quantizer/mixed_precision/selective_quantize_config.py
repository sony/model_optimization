# Copyright 2021 Sony Semiconductor Israel, Inc. All rights reserved.
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
# As from Tensorflow 2.6, keras is a separate package and some classes should be imported differently.
from model_compression_toolkit.core.common.quantization.candidate_node_quantization_config import \
    CandidateNodeQuantizationConfig
from model_compression_toolkit.core.keras.quantizer.mixed_precision.selective_activation_quantizer import \
    SelectiveActivationQuantizer
from packaging import version
from model_compression_toolkit.core.common.logger import Logger

if version.parse(tf.__version__) < version.parse("2.6"):
    from tensorflow.python.keras.layers import Layer  # pragma: no cover
else:
    from keras.engine.base_layer import Layer
from tensorflow.python.training.tracking.data_structures import ListWrapper
from tensorflow_model_optimization.python.core.quantization.keras.quantize_config import QuantizeConfig

from model_compression_toolkit.core.keras.quantizer.mixed_precision.selective_weights_quantizer import SelectiveWeightsQuantizer


class SelectiveQuantizeConfig(QuantizeConfig):
    """
    SelectiveQuantizeConfig to use as a QuantizeCong for layers that are wrapped
    for MP models. SelectiveQuantizeConfig holds a SelectiveWeightsQuantizer and uses
    it to use quantized weight from a set of quantized weights (each one of the
    quantized weights was quantized with different bitwidth).
    At any given time, the SelectiveQuantizeConfig uses only one quantized weight
    according to an "active" index - the index of a candidate weight quantization configuration
    from a list of candidates that was passed to the SelectiveQuantizeConfig when it was initialized.
    The "active" index can be configured as part of the SelectiveQuantizeConfig's API,
    so a different quantized weight can be used in another time.
    """

    def __init__(self,
                 node_q_cfg: List[CandidateNodeQuantizationConfig],
                 float_weights: List[np.ndarray] = None,
                 weight_attrs: List[str] = None,
                 max_candidate_idx: int = 0):
        """
        Init a SelectiveQuantizeConfig instance.

        Args:
            weight_attrs: Attributes of the layer's weights to quantize, the
            SelectiveQuantizeConfig is attached to.
            float_weights: Float weights of the layer, the SelectiveQuantizeConfig is attached to.
            node_q_cfg: Candidates quantization config the node has (the node from which
            we built the layer that is attached to SelectiveQuantizeConfig).
            max_candidate_idx: Index of the node's candidate that has the maximal bitwidth (must exist absolute max).
        """
        # Make sure the candidates configurations arrived in a descending order.
        curmax = (np.inf, np.inf)
        n_candidate_bits = [(x.weights_quantization_cfg.weights_n_bits, x.activation_quantization_cfg.activation_n_bits)
                            for x in node_q_cfg]
        for candidate_bits in n_candidate_bits:
            assert candidate_bits < curmax
            curmax = candidate_bits

        self.weight_attrs = weight_attrs
        self.float_weights = float_weights

        assert len(node_q_cfg) > 0, 'SelectiveQuantizeConfig has to receive' \
                                            'at least one quantization configuration'
        assert (not weight_attrs and not float_weights) or len(weight_attrs) == len(float_weights)

        for qc in node_q_cfg:
            assert qc.weights_quantization_cfg.enable_weights_quantization == \
                   node_q_cfg[0].weights_quantization_cfg.enable_weights_quantization \
                   and qc.activation_quantization_cfg.enable_activation_quantization == \
                   node_q_cfg[0].activation_quantization_cfg.enable_activation_quantization, \
                "Candidates with different weights/activation enabled properties is currently not supported"

        self.node_q_cfg = node_q_cfg
        self.enable_weights_quantization = node_q_cfg[0].weights_quantization_cfg.enable_weights_quantization
        self.enable_activation_quantization = node_q_cfg[0].activation_quantization_cfg.enable_activation_quantization

        # Initialize a SelectiveWeightsQuantizer for each weight that should be quantized.
        self.weight_quantizers = []
        if self.enable_weights_quantization:
            self.weight_quantizers = [SelectiveWeightsQuantizer(node_q_cfg,
                                                                float_weight=float_weight,
                                                                max_candidate_idx=max_candidate_idx) for float_weight
                                      in float_weights]

        self.activation_selective_quantizer = None if not self.enable_activation_quantization else \
            SelectiveActivationQuantizer(node_q_cfg, max_candidate_idx=max_candidate_idx)

    def set_bit_width_index(self,
                            index: int,
                            attr: str = None):
        """
        Change the "active" bitwidth index the SelectiveQuantizeConfig uses, so
        a different quantized weight and activation will be used.
        If attr is passed, only the quantizer that was created for this attribute will be configured.
        Otherwise, all quantizers the SelectiveQuantizeConfig holds will be configured
        using the passed index.
        Args:
            index: Bitwidth index to use.
            attr: Name of the layer's weights attribute to configure its corresponding quantizer.
        """

        self.set_weights_bit_width_index(index, attr)
        self.set_activation_bit_width_index(index)

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

        if self.enable_weights_quantization:
            if attr is None:  # set bit width to all weights of the layer
                for q in self.weight_quantizers:
                    q.set_active_quantization_config_index(index)
            else:  # set bit width to a specific attribute
                i = self.weight_attrs.index(attr)
                q = self.weight_quantizers[i]
                q.set_active_quantization_config_index(index)

    def set_activation_bit_width_index(self,
                                       index: int):
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
        if self.enable_activation_quantization:
            self.activation_selective_quantizer.set_active_quantization_config_index(index)

    def get_weights_and_quantizers(self, layer: Layer) -> List[Tuple[Tensor, Any]]:
        """
        Get a list of tuples with weights and the weights quantizers.
        The layer's attributes are used to get the weights.

        Args:
            layer: The layer the SelectiveQuantizeConfig is attached to when is wrapped.

        Returns:
            List of tuples of the layer's weights and the weights quantizers.
        """
        return [] if not self.enable_weights_quantization else \
            [(getattr(layer, self.weight_attrs[i]), self.weight_quantizers[i]) for i in range(len(self.weight_attrs))]

    def get_activations_and_quantizers(self, layer: Layer) -> list:
        # For configurable activations we use get_output_quantizers,
        # Therefore, we do not need to implement this method.
        return []

    def set_quantize_weights(self, layer: Layer, quantize_weights: List[Tensor]):
        """
        Set the layer weights with new passed weights.
        Args:
            layer: Layer to set its attributes.
            quantize_weights: Quantized weights to set as new weights.

        """
        if self.enable_weights_quantization:
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

    def set_quantize_activations(self, layer, quantize_activations: ListWrapper):
        pass

    def get_output_quantizers(self, layer: Layer) -> list:
        return [] if not self.enable_activation_quantization else [self.activation_selective_quantizer]

    def get_config(self) -> Dict[str, Any]:
        """
        Returns: The SelectiveQuantizeConfig configuration.
        """

        return {
            'weight_attrs': self.weight_attrs,
            'float_weights': self.float_weights,
            'node_q_cfg': self.node_q_cfg
        }
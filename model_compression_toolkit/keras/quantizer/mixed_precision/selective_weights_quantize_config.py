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

import numpy as np
from tensorflow import Tensor
import tensorflow as tf
# As from Tensorflow 2.6, keras is a separate package and some classes should be imported differently.
from model_compression_toolkit.common.quantization.candidate_node_quantization_config import \
    CandidateNodeQuantizationConfig

if tf.__version__ < "2.6":
    from tensorflow.python.keras.layers import Layer
else:
    from keras.engine.base_layer import Layer
from tensorflow.python.training.tracking.data_structures import ListWrapper
from tensorflow_model_optimization.python.core.quantization.keras.quantize_config import QuantizeConfig

from model_compression_toolkit.keras.quantizer.mixed_precision.selective_quantizer import SelectiveQuantizer


class SelectiveWeightsQuantizeConfig(QuantizeConfig):
    """
    SelectiveWeightsQuantizeConfig to use as a QuantizeCong for layers that are wrapped
    for MP models. SelectiveWeightsQuantizeConfig holds a SelectiveQuantizer and uses
    it to use quantized weight from a set of quantized weights (each one of the
    quantized weights was quantized with different bitwidth).
    At any given time, the SelectiveWeightsQuantizeConfig uses only one quantized weight
    according to an "active" index - the index of a candidate weight quantization configuration
    from a list of candidates that was passed to the SelectiveWeightsQuantizeConfig when it was initialized.
    The "active" index can be configured as part of the SelectiveWeightsQuantizeConfig's API,
    so a different quantized weight can be used in another time.
    """

    def __init__(self,
                 weight_attrs: List[str],
                 float_weights: List[np.ndarray],
                 node_q_cfg: List[CandidateNodeQuantizationConfig]):
        """
        Init a SelectiveWeightsQuantizeConfig instance.

        Args:
            weight_attrs: Attributes of the layer's weights to quantize, the
            SelectiveWeightsQuantizeConfig is attached to.
            float_weights: Float weights of the layer, the SelectiveWeightsQuantizeConfig is attached to.
            node_q_cfg: Candidates quantization config the node has (the node from which
            we built the layer that is attached to SelectiveWeightsQuantizeConfig).
        """
        # Make sure the candidates configurations arrived in a descending order.
        curmax = np.inf
        for n_candidate in node_q_cfg:
            assert n_candidate.weights_quantization_cfg.weights_n_bits < curmax
            curmax = n_candidate.weights_quantization_cfg.weights_n_bits

        self.weight_attrs = weight_attrs
        assert len(node_q_cfg) > 0, 'SelectiveWeightsQuantizeConfig has to receive' \
                                            'at least one weight quantization configuration'
        assert len(weight_attrs) == len(float_weights)

        self.node_q_cfg = node_q_cfg

        # Initialize a SelectiveQuantizer for each weight that should be quantized.
        self.weight_quantizers = [SelectiveQuantizer(node_q_cfg,
                                                     float_weight=float_weight) for float_weight in float_weights]

    def set_bit_width_index(self,
                            index: int,
                            attr: str=None):
        """
        Change the "active" bitwidth index the SelectiveWeightsQuantizeConfig uses, so
        a different quantized weight will be used.
        If attr is passed, only the quantizer that was created for this attribute will be configured.
        Otherwise, all quantizers the SelectiveWeightsQuantizeConfig holds will be configured
        using the passed index.

        Args:
            index: Bitwidth index to use.
            attr: Name of the layer's attribute to configure its corresponding quantizer.

        """

        if attr is None:  # set bit width to all weights of the layer
            for q in self.weight_quantizers:
                q.set_active_quantization_config_index(index)
        else:  # set bit width to a specific selectivequantizer
            i = self.weight_attrs.index(attr)
            q = self.weight_quantizers[i]
            q.set_active_quantization_config_index(index)

    def get_weights_and_quantizers(self, layer: Layer) -> List[Tuple[Tensor, Any]]:
        """
        Get a list of tuples with weights and the weights quantizers.
        The layer's attributes are used to get the weights.
        Args:
            layer: The layer the SelectiveWeightsQuantizeConfig is attached to when is wrapped.

        Returns:
            List of tuples of the layer's weights and the weights quantizers.
        """

        return [(getattr(layer, self.weight_attrs[i]), self.weight_quantizers[i])
                for i in range(len(self.weight_attrs))]

    def get_activations_and_quantizers(self, layer: Layer) -> list:
        # This QuantizeConfig is for quantizing weights only, so no
        # implementation is needed for activation quantization.
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
                    layer.name, len(quantize_weights), len(self.weight_attrs)))

        for weight_attr, weight in zip(self.weight_attrs, quantize_weights):
            current_weight = getattr(layer, weight_attr)
            if current_weight.shape != weight.shape:
                raise ValueError('Existing layer weight shape {} is incompatible with'
                                 'provided weight shape {}'.format(
                    current_weight.shape, weight.shape))

            setattr(layer, weight_attr, weight)

    def set_quantize_activations(self, layer, quantize_activations: ListWrapper):
        # This QuantizeConfig is for quantizing weights only, so no
        # implementation is needed for activation quantization.
        pass

    def get_output_quantizers(self, layer: Layer) -> list:
        # This QuantizeConfig is for quantizing weights only, so no
        # implementation is needed for activation quantization.
        return []

    def get_config(self) -> Dict[str, Any]:
        """
        Returns: The SelectiveWeightsQuantizeConfig configuration.
        """

        return {
            'weight_attrs': self.weight_attrs,
            'weights_candidates_quantization_configs': self.weights_candidates_quantization_configs
        }

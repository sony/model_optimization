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
from mct_quantizers import KerasQuantizationWrapper, KerasActivationQuantizationHolder
from tensorflow.python.layers.base import Layer

from model_compression_toolkit.core.keras.mixed_precision.configurable_activation_quantizer import \
    ConfigurableActivationQuantizer
from model_compression_toolkit.core.keras.mixed_precision.configurable_weights_quantizer import \
    ConfigurableWeightsQuantizer


def set_layer_to_bitwidth(quantization_layer: Layer,
                          bitwidth_idx: int):
    """
    Configures a layer's configurable quantizer to work with a different bit-width.
    The bit-width_idx is the index of the actual quantizer the quantizer object in the quantization_layer wraps/holds.

    Args:
        quantization_layer: Layer to change its bit-width.
        bitwidth_idx: Index of the bit-width the layer should work with.
    """

    if isinstance(quantization_layer, KerasQuantizationWrapper):
        for weight_attr, quantizer in quantization_layer.weights_quantizers.items():
            if isinstance(quantizer, ConfigurableWeightsQuantizer):
                # Setting bitwidth only for configurable layers. There might be wrapped layers that aren't configurable,
                # for instance, if only activations are quantized with mixed precision and weights are quantized with
                # fixed precision
                quantizer.set_weights_bit_width_index(bitwidth_idx, weight_attr)

    if isinstance(quantization_layer, KerasActivationQuantizationHolder):
        if isinstance(quantization_layer.activation_holder_quantizer, ConfigurableActivationQuantizer):
            # Setting bitwidth only for configurable layers. There might be activation layers that isn't configurable,
            # for instance, if only weights are quantized with mixed precision and activation are quantized with
            # fixed precision
            quantization_layer.activation_holder_quantizer.set_active_quantization_config_index(bitwidth_idx)

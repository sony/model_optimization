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
import typing
from typing import Any, Optional

if typing.TYPE_CHECKING:    # pragma: no cover
    from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation


def set_activation_quant_layer_to_bitwidth(quantization_layer: Any,
                                           bitwidth_idx: Optional[int],
                                           fw_impl: 'FrameworkImplementation'):
    """
    Configures a layer's configurable activation quantizer to work with a different bit-width.
    The bit-width_idx is the index of the actual quantizer the quantizer object in the quantization_layer wraps/holds.

    Args:
        quantization_layer: Layer to change its bit-width.
        bitwidth_idx: Index of the bit-width the layer should work with, or None to disable quantization.
        fw_impl: framework implementation object.
    """
    assert isinstance(quantization_layer, fw_impl.activation_quant_layer_cls)
    assert isinstance(quantization_layer.activation_holder_quantizer, fw_impl.configurable_activation_quantizer_cls)
    quantization_layer.activation_holder_quantizer.set_active_activation_quantizer(bitwidth_idx)


def set_weights_quant_layer_to_bitwidth(quantization_layer: Any,
                                        bitwidth_idx: Optional[int],
                                        fw_impl: 'FrameworkImplementation'):
    """
    Configures a layer's configurable weights quantizer to work with a different bit-width.
    The bit-width_idx is the index of the actual quantizer the quantizer object in the quantization_layer wraps/holds.

    Args:
        quantization_layer: Layer to change its bit-width.
        bitwidth_idx: Index of the bit-width the layer should work with, or None to disable quantization.
        fw_impl: framework implementation object.
    """
    assert isinstance(quantization_layer, fw_impl.weights_quant_layer_cls)
    configurable_quantizers = [q for q in quantization_layer.weights_quantizers.values()
                               if isinstance(q, fw_impl.configurable_weights_quantizer_cls)]
    assert configurable_quantizers
    for quantizer in configurable_quantizers:
        quantizer.set_weights_bit_width_index(bitwidth_idx)

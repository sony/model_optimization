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
from tensorflow.python.layers.base import Layer
from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper

from model_compression_toolkit.core.keras.quantizer.mixed_precision.selective_quantize_config import \
    SelectiveQuantizeConfig


def set_layer_to_bitwidth(wrapped_layer: Layer,
                          bitwidth_idx: int):
    """
    Configure a layer (which is wrapped in a QuantizeWrapper and holds a
    SelectiveQuantizeConfig  in its quantize_config) to work with a different bit-width.
    The bit-width_idx is the index of the quantized-weights the quantizer in the SelectiveQuantizeConfig holds.

    Args:
        wrapped_layer: Layer to change its bit-width.
        bitwidth_idx: Index of the bit-width the layer should work with.
    """
    assert isinstance(wrapped_layer, QuantizeWrapper) and isinstance(wrapped_layer.quantize_config,
                                                                     SelectiveQuantizeConfig)
    # Configure the quantize_config to use a different bit-width
    # (in practice, to use a different already quantized kernel).
    wrapped_layer.quantize_config.set_bit_width_index(bitwidth_idx)

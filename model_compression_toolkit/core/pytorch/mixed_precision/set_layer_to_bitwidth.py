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
from torch.nn import Module

from model_compression_toolkit.core.pytorch.mixed_precision.mixed_precision_wrapper import PytorchMixedPrecisionWrapper


def set_layer_to_bitwidth(wrapped_layer: Module,
                          bitwidth_idx: int):
    """
    Configure a layer (which is wrapped in a PytorchMixedPrecisionWrapper and holds a model's layer (nn.Module))
    to work with a different bit-width.
    The bitwidth_idx is the index of the quantized-weights the quantizer in the PytorchMixedPrecisionWrapper holds.

    Args:
        wrapped_layer: Layer to change its bit-width.
        bitwidth_idx: Index of the bit-width the layer should work with.
    """
    assert isinstance(wrapped_layer, PytorchMixedPrecisionWrapper)
    # Configure the quantize_config to use a different bitwidth
    # (in practice, to use a different already quantized kernel).
    wrapped_layer.set_active_weights(bitwidth_idx)
    wrapped_layer.set_active_activation_quantizer(bitwidth_idx)

# Copyright 2025 Sony Semiconductor Israel, Inc. All rights reserved.
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
import pytest
from unittest.mock import Mock

from model_compression_toolkit.exporter.model_wrapper.pytorch.builder.fully_quantized_model_builder import get_activation_quantizer_holder
from mct_quantizers import PytorchActivationQuantizationHolder, PytorchPreservingActivationQuantizationHolder

def test_get_activation_quantizer_holder(fw_impl_mock):

    activation_quantizers = Mock()
    activation_quantizers.num_bits = 8
    activation_quantizers.signed = False
    activation_quantizers.threshold_np = 4.0
    fw_impl_mock.get_inferable_quantizers.return_value = (None, [activation_quantizers])
    activation_quantization_holder = get_activation_quantizer_holder(node=Mock(), holder_type=PytorchActivationQuantizationHolder, fw_impl=fw_impl_mock)

    assert isinstance(activation_quantization_holder, PytorchActivationQuantizationHolder)
    assert activation_quantization_holder.activation_holder_quantizer.num_bits == 8
    assert activation_quantization_holder.activation_holder_quantizer.signed == False
    assert activation_quantization_holder.activation_holder_quantizer.threshold_np == 4.0

def test_get_preserving_activation_quantizer_holder(fw_impl_mock):

    activation_quantizers = Mock()
    activation_quantizers.num_bits = 4
    activation_quantizers.signed = True
    activation_quantizers.threshold_np = 8.0
    fw_impl_mock.get_inferable_quantizers.return_value = (None, [activation_quantizers])
    activation_quantization_holder = get_activation_quantizer_holder(node=Mock(), holder_type=PytorchPreservingActivationQuantizationHolder, fw_impl=fw_impl_mock)

    assert isinstance(activation_quantization_holder, PytorchPreservingActivationQuantizationHolder)
    assert activation_quantization_holder.quantization_bypass == True
    assert activation_quantization_holder.activation_holder_quantizer.num_bits == 4
    assert activation_quantization_holder.activation_holder_quantizer.signed == True
    assert activation_quantization_holder.activation_holder_quantizer.threshold_np == 8.0
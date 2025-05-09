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
import torch
from mct_quantizers import PytorchActivationQuantizationHolder, PytorchQuantizationWrapper
from mct_quantizers.pytorch.quantizer_utils import to_torch_tensor

from model_compression_toolkit.core.common.mixed_precision.set_layer_to_bitwidth import \
    set_activation_quant_layer_to_bitwidth, set_weights_quant_layer_to_bitwidth
from model_compression_toolkit.core.pytorch.constants import KERNEL
from model_compression_toolkit.core.pytorch.mixed_precision.configurable_activation_quantizer import \
    ConfigurableActivationQuantizer
from model_compression_toolkit.core.pytorch.mixed_precision.configurable_weights_quantizer import \
    ConfigurableWeightsQuantizer
from model_compression_toolkit.core.pytorch.pytorch_device_config import get_working_device
from tests_pytest._test_util.graph_builder_utils import build_nbits_qc
from tests_pytest.pytorch_tests.torch_test_util.torch_test_mixin import TorchFwMixin


class TestConfigureQLayer(TorchFwMixin):
    @pytest.mark.parametrize('ind', [None, 0, 1, 2])
    def test_configure_activation(self, ind):
        """ Test correct activation quantizer is set and applied. """
        def quant_fn(nbits, *args, **kwargs):
            return lambda x: x*nbits
        abits = [8, 4, 2]
        quantizer = ConfigurableActivationQuantizer(node_q_cfg=[
            build_nbits_qc(abit, activation_quantization_fn=quant_fn) for abit in abits
        ])
        layer = PytorchActivationQuantizationHolder(quantizer)
        set_activation_quant_layer_to_bitwidth(layer, ind, self.fw_impl)
        assert quantizer.active_quantization_config_index == ind
        x = torch.rand(100)
        y = layer(x)
        if ind is None:
            assert torch.equal(x, y)
        else:
            assert torch.allclose(x*abits[ind], y)

    @pytest.mark.parametrize('ind', [None, 0, 1, 2])
    def test_configure_weights(self, ind):
        """ Test correct weights quantizer is set and applied. """
        inner_layer = torch.nn.Conv2d(3, 8, kernel_size=5).to(get_working_device())
        orig_weight = inner_layer.weight.clone()
        orig_bias = inner_layer.bias.clone()

        wbits = [8, 4, 2]
        qcs = [build_nbits_qc(w_attr={KERNEL: (wbit, True)}) for wbit in wbits]
        for qc in qcs:
            attr_cfg = qc.weights_quantization_cfg.get_attr_config(KERNEL)
            attr_cfg.weights_channels_axis = (0,)
            attr_cfg.weights_quantization_fn = lambda x, nbits, *args: x*nbits

        quantizer = ConfigurableWeightsQuantizer(
            node_q_cfg=qcs,
            float_weights=inner_layer.weight,
            kernel_attr=KERNEL
        )
        layer = PytorchQuantizationWrapper(inner_layer, {KERNEL: quantizer})

        set_weights_quant_layer_to_bitwidth(layer, ind, self.fw_impl)

        assert quantizer.active_quantization_config_index == ind
        x = to_torch_tensor(torch.rand((1, 3, 16, 16), dtype=torch.float32))
        y = layer(x)
        # check that correct quantizer was indeed applied by applying quantization function to kernel manually
        # and comparing the outputs
        ref_layer = torch.nn.Conv2d(3, 8, kernel_size=5).to(get_working_device())
        ref_layer.weight.data = orig_weight.data
        ref_layer.bias.data = orig_bias.data
        if ind is not None:
            ref_layer.weight.data *= wbits[ind]
        y_ref = ref_layer(x)
        assert torch.allclose(y, y_ref)

        # check that can be configured and run again
        set_weights_quant_layer_to_bitwidth(layer, 1, self.fw_impl)
        layer(x)

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
import keras
import numpy as np
import pytest
from mct_quantizers import KerasActivationQuantizationHolder, KerasQuantizationWrapper

from model_compression_toolkit.core.common.mixed_precision.set_layer_to_bitwidth import \
    set_activation_quant_layer_to_bitwidth, set_weights_quant_layer_to_bitwidth
from model_compression_toolkit.core.keras.constants import KERNEL
from model_compression_toolkit.core.keras.mixed_precision.configurable_activation_quantizer import \
    ConfigurableActivationQuantizer
from model_compression_toolkit.core.keras.mixed_precision.configurable_weights_quantizer import \
    ConfigurableWeightsQuantizer
from tests_pytest._test_util.graph_builder_utils import build_nbits_qc
from tests_pytest.keras_tests.keras_test_util.keras_test_mixin import KerasFwMixin


class TestConfigureQLayer(KerasFwMixin):
    @pytest.mark.parametrize('ind', [None, 0, 1, 2])
    def test_configure_activation(self, ind):
        """ Test correct activation quantizer is set and applied. """
        def quant_fn(nbits, *args, **kwargs):
            return lambda x: x*nbits
        abits = [8, 4, 2]
        quantizer = ConfigurableActivationQuantizer(node_q_cfg=[
            build_nbits_qc(abit, activation_quantization_fn=quant_fn) for abit in abits
        ])
        layer = KerasActivationQuantizationHolder(quantizer)
        set_activation_quant_layer_to_bitwidth(layer, ind, self.fw_impl)
        assert quantizer.active_quantization_config_index == ind
        x = np.random.rand(100)
        y = layer(x)
        if ind is None:
            assert np.allclose(x, y)
        else:
            assert np.allclose(x*abits[ind], y)

    @pytest.mark.parametrize('ind', [None, 0, 1, 2])
    def test_configure_weights(self, ind):
        """ Test correct weights quantizer is set and applied. """
        inp = keras.layers.Input(shape=(16, 16, 3))
        out = keras.layers.Conv2D(8, kernel_size=5)(inp)
        model = keras.Model(inp, out)
        inner_layer = model.layers[1]
        orig_weight = inner_layer.kernel.numpy()
        orig_bias = inner_layer.bias.numpy()

        wbits = [8, 4, 2]
        qcs = [build_nbits_qc(w_attr={KERNEL: (wbit, True)}) for wbit in wbits]
        for qc in qcs:
            attr_cfg = qc.weights_quantization_cfg.get_attr_config(KERNEL)
            attr_cfg.weights_channels_axis = (0,)
            attr_cfg.weights_quantization_fn = lambda x, nbits, *args: x*nbits
        quantizer = ConfigurableWeightsQuantizer(
            node_q_cfg=qcs,
            float_weights=inner_layer.kernel.numpy(),
            kernel_attr=KERNEL
        )
        layer = KerasQuantizationWrapper(inner_layer, {KERNEL: quantizer})

        set_weights_quant_layer_to_bitwidth(layer, ind, self.fw_impl)

        assert quantizer.active_quantization_config_index == ind
        x = np.random.rand(1, 16, 16, 3).astype(np.float32)
        y = layer(x)
        # check that correct quantizer was indeed applied by applying quantization function to kernel manually
        # and comparing the outputs
        ref_inp = keras.layers.Input(shape=(16, 16, 3))
        weight = orig_weight
        if ind is not None:
            weight *= wbits[ind]
        ref_out = keras.layers.Conv2D(8, kernel_size=5, kernel_initializer=keras.initializers.Constant(weight),
                                      bias_initializer=keras.initializers.Constant(orig_bias))(ref_inp)
        ref_model = keras.Model(ref_inp, ref_out)
        ref_layer = ref_model.layers[1]
        y_ref = ref_layer(x)
        assert np.allclose(y, y_ref)

        # check that can be configured and run again
        set_weights_quant_layer_to_bitwidth(layer, 1, self.fw_impl)
        layer(x)

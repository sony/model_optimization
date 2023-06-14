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
import keras
from keras import Input
from keras.layers import Conv2D, TFOpLambda
import numpy as np
import tensorflow as tf

from mct_quantizers import QuantizationMethod
from mct_quantizers.keras.quantizers import WeightsLUTSymmetricInferableQuantizer
from tests.common_tests.helpers.generate_test_tp_model import generate_test_tp_model
from model_compression_toolkit.target_platform_capabilities.tpc_models.default_tpc.latest import generate_keras_tpc
from tests.keras_tests.exporter_tests.keras_fake_quant.keras_fake_quant_exporter_base_test import \
    KerasFakeQuantExporterBaseTest, get_minmax_from_qparams

class TestConv2DKerasFQExporter(KerasFakeQuantExporterBaseTest):

    def get_input_shape(self):
        return [(30, 30, 3)]

    def get_tpc(self):
        tp = generate_test_tp_model({'weights_n_bits': 2,
                                     'activation_n_bits': 2})
        return generate_keras_tpc(name="test_conv2d_2bit_fq_weight", tp_model=tp)

    def get_model(self):
        inputs = Input(shape=self.get_input_shape()[0])
        x = Conv2D(6, 20)(inputs)
        model = keras.Model(inputs=inputs, outputs=x)
        return model

    def run_checks(self):
        assert len(self.loaded_model.layers)==4
        assert isinstance(self.loaded_model.layers[1], TFOpLambda)
        assert self.loaded_model.layers[1].function==tf.quantization.fake_quant_with_min_max_vars
        assert isinstance(self.loaded_model.layers[2], Conv2D)
        assert isinstance(self.loaded_model.layers[3], TFOpLambda)
        assert self.loaded_model.layers[3].function == tf.quantization.fake_quant_with_min_max_vars

        # Check conv weights exported correctly
        assert np.max(np.abs(self.loaded_model.layers[2].kernel - self.exportable_model.layers[2].get_quantized_weights()['kernel']))==0
        assert np.all(self.exportable_model.layers[2].layer.bias == self.loaded_model.layers[2].bias)
        for i in range(6):
            assert len(np.unique(self.loaded_model.layers[2].kernel[:, :, :, i]))<=2**2

        # Check conv activation qparams are exported correctly
        conv_qparams = self.exportable_model.layers[3].get_config()['activation_holder_quantizer']['config']
        assert conv_qparams['num_bits'] == 2
        _min, _max = get_minmax_from_qparams(conv_qparams)
        assert _min == self.loaded_model.layers[3].inbound_nodes[0].call_kwargs['min']
        assert _max == self.loaded_model.layers[3].inbound_nodes[0].call_kwargs['max']



class TestConv2DReusedKerasFQExporter(TestConv2DKerasFQExporter):

    def get_model(self):
        conv=Conv2D(3, 3)
        inputs = Input(shape=self.get_input_shape()[0])
        x = conv(inputs)
        x = conv(x)
        model = keras.Model(inputs=inputs, outputs=x)
        return model

    def run_checks(self):
        assert len(self.loaded_model.layers)==5
        assert isinstance(self.loaded_model.layers[1], TFOpLambda)
        assert self.loaded_model.layers[1].function==tf.quantization.fake_quant_with_min_max_vars
        assert isinstance(self.loaded_model.layers[2], Conv2D)
        assert isinstance(self.loaded_model.layers[3], TFOpLambda)
        assert self.loaded_model.layers[3].function == tf.quantization.fake_quant_with_min_max_vars
        assert isinstance(self.loaded_model.layers[4], TFOpLambda)
        assert self.loaded_model.layers[4].function == tf.quantization.fake_quant_with_min_max_vars

        # Check conv weights exported correctly
        assert np.max(np.abs(self.loaded_model.layers[2].kernel - self.exportable_model.layers[2].get_quantized_weights()['kernel']))==0
        assert np.all(self.exportable_model.layers[2].layer.bias == self.loaded_model.layers[2].bias)
        for i in range(3):
            assert len(np.unique(self.loaded_model.layers[2].kernel[:, :, :, i]))<=2**2

        # Reused layers have no 'input_shape' attribute since they have multiple inputs with different input shapes
        assert not hasattr(self.loaded_model.layers[2], 'input_shape')

        # Check conv activation qparams are exported correctly
        conv_qparams = self.exportable_model.layers[3].get_config()['activation_holder_quantizer']['config']
        assert conv_qparams['num_bits'] == 2
        _min, _max = get_minmax_from_qparams(conv_qparams)
        assert _min == self.loaded_model.layers[3].inbound_nodes[0].call_kwargs['min']
        assert _max == self.loaded_model.layers[3].inbound_nodes[0].call_kwargs['max']


class TestCon2DWeightsLUTKerasFQExporter(KerasFakeQuantExporterBaseTest):

    def get_input_shape(self):
        return [(30, 30, 3)]

    def get_tpc(self):
        tp = generate_test_tp_model({'weights_n_bits': 2,
                                     'weights_quantization_method': QuantizationMethod.LUT_SYM_QUANTIZER})
        return generate_keras_tpc(name="test_conv2d_2bit_lut_fq_weight", tp_model=tp)

    def get_model(self):
        inputs = Input(shape=self.get_input_shape()[0])
        x = Conv2D(6, 20)(inputs)
        model = keras.Model(inputs=inputs, outputs=x)
        return model

    def run_checks(self):
        assert isinstance(self.exportable_model.layers[2].weights_quantizers['kernel'], WeightsLUTSymmetricInferableQuantizer)
        # Check conv weights exported correctly
        assert np.max(np.abs(self.loaded_model.layers[2].kernel - self.exportable_model.layers[2].get_quantized_weights()['kernel']))==0
        assert np.all(self.exportable_model.layers[2].layer.bias == self.loaded_model.layers[2].bias)
        for i in range(6):
            assert len(np.unique(self.loaded_model.layers[2].kernel[:, :, :, i]))<=2**2




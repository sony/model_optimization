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
import copy
import os

import tensorflow as tf
import numpy as np

from tests.keras_tests.tpc_keras import get_tpc
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
import model_compression_toolkit as mct
from model_compression_toolkit import quantizers_infrastructure as qi
import os
from model_compression_toolkit.exporter.model_wrapper.keras.builder.node_to_quantizer import \
    QUANTIZATION_METHOD_2_ACTIVATION_QUANTIZER, QUANTIZATION_METHOD_2_WEIGHTS_QUANTIZER
from model_compression_toolkit.qat.keras.quantizer.quantization_builder import METHOD2ACTQUANTIZER, \
    METHOD2WEIGHTQUANTIZER
from model_compression_toolkit.core.keras.default_framework_info import KERNEL

keras = tf.keras
layers = keras.layers


class QuantizationAwareTrainingTest(BaseKerasFeatureNetworkTest):
    def __init__(self, unit_test, layer, weight_bits=2, activation_bits=4, finalize=False,
                 weights_quantization_method=mct.target_platform.QuantizationMethod.POWER_OF_TWO,
                 activation_quantization_method=mct.target_platform.QuantizationMethod.POWER_OF_TWO,
                 test_loading=False):
        self.layer = layer
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        assert finalize is False, 'MCT QAT Finalize is disabled until exporter is fully supported'
        self.finalize = finalize
        self.weights_quantization_method = weights_quantization_method
        self.activation_quantization_method = activation_quantization_method
        self.test_loading = test_loading
        super().__init__(unit_test)

    def get_tpc(self):
        return get_tpc("QAT_test", weight_bits=self.weight_bits, activation_bits=self.activation_bits,
                       weights_quantization_method=self.weights_quantization_method,
                       activation_quantization_method=self.activation_quantization_method)

    def run_test(self, experimental_facade=False, **kwargs):
        model_float = self.create_networks()
        ptq_model, quantization_info, custom_objects = mct.keras_quantization_aware_training_init(model_float,
                                                                                                  self.representative_data_gen,
                                                                                                  fw_info=self.get_fw_info(),
                                                                                                  target_platform_capabilities=self.get_tpc())

        ptq_model2 = None
        if self.test_loading:
            ptq_model.save('qat2model.h5')
            ptq_model2 = mct.keras_load_quantized_model('qat2model.h5')
            os.remove('qat2model.h5')

        if self.finalize:
            ptq_model = mct.keras_quantization_aware_training_finalize(ptq_model)

        self.compare(ptq_model,
                     model_float,
                     ptq_model2,
                     input_x=self.representative_data_gen(),
                     quantization_info=quantization_info)

    def compare(self, quantized_model, float_model, loaded_model, input_x=None, quantization_info=None):
        if self.test_loading:
            for lo, ll in zip(quantized_model.layers, loaded_model.layers):
                if isinstance(ll, qi.KerasQuantizationWrapper):
                    self.unit_test.assertTrue(isinstance(lo, qi.KerasQuantizationWrapper))
                if isinstance(lo, qi.KerasQuantizationWrapper):
                    self.unit_test.assertTrue(isinstance(ll, qi.KerasQuantizationWrapper))
                    for w_ll, w_lo in zip(ll.weights, lo.weights):
                        self.unit_test.assertTrue(np.all(w_ll.numpy() == w_lo.numpy()))

        if self.finalize:
            self.unit_test.assertTrue(isinstance(quantized_model.layers[2], type(self.layer)))
        else:
            self.unit_test.assertTrue(isinstance(quantized_model.layers[2].layer, type(self.layer)))

            # TODO:refactor test
            # _, qconfig = quantized_model.layers[2].quantize_config.get_weights_and_quantizers(quantized_model.layers[2].layer)[0]
            # self.unit_test.assertTrue(qconfig.num_bits == self.weight_bits)


class QuantizationAwareTrainingQuantizersTest(QuantizationAwareTrainingTest):

    def __init__(self, unit_test, weight_bits=8, activation_bits=4, finalize=False):
        super().__init__(unit_test, layers.DepthwiseConv2D(5, activation='relu'),
                         weight_bits=weight_bits, activation_bits=activation_bits, finalize=finalize)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        outputs = self.layer(inputs)
        w = np.arange(5 * 5 * 3, dtype=np.float32).reshape((3, 5, 5, 1)).transpose((1, 2, 0, 3))
        # Add LSB to verify the correct threshold is chosen and applied per channel
        w[0, 0, :, 0] += np.array([0.25, 0.5, 0.])
        self.layer.weights[0].assign(w)
        return keras.Model(inputs=inputs, outputs=outputs)

    def compare(self, quantized_model, float_model, loaded_model, input_x=None, quantization_info=None):
        if self.finalize:
            self.unit_test.assertTrue(isinstance(quantized_model.layers[2], layers.DepthwiseConv2D))
            dw_weight = float_model.layers[1].weights[0].numpy()
            quantized_dw_weight = quantized_model.layers[2].weights[0].numpy()
        else:
            self.unit_test.assertTrue(isinstance(quantized_model.layers[2].layer, layers.DepthwiseConv2D))
            for name, quantizer in quantized_model.layers[2].weights_quantizers.items():
                w_select = [w for w in float_model.layers[1].weights if name + ":0" in w.name]
                if len(w_select) != 1:
                    raise Exception()
                dw_weight = w_select[0]
                quantized_dw_weight = quantizer(dw_weight, False)
        self.unit_test.assertTrue(np.all(dw_weight == quantized_dw_weight))


class QATWrappersTest(BaseKerasFeatureNetworkTest):
    def __init__(self, unit_test, layer, weight_bits=2, activation_bits=4, finalize=True,
                 weights_quantization_method=mct.target_platform.QuantizationMethod.POWER_OF_TWO,
                 activation_quantization_method=mct.target_platform.QuantizationMethod.POWER_OF_TWO,
                 test_loading=False):
        self.layer = layer
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.finalize = finalize
        self.weights_quantization_method = weights_quantization_method
        self.activation_quantization_method = activation_quantization_method
        self.test_loading = test_loading
        super().__init__(unit_test)

    def get_tpc(self):
        return get_tpc("QAT_wrappers_test", weight_bits=self.weight_bits, activation_bits=self.activation_bits,
                       weights_quantization_method=self.weights_quantization_method,
                       activation_quantization_method=self.activation_quantization_method)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        outputs = self.layer(inputs)
        return keras.Model(inputs=inputs, outputs=outputs)

    def run_test(self, experimental_facade=False, **kwargs):
        model_float = self.create_networks()
        ptq_model, quantization_info, custom_objects = mct.keras_quantization_aware_training_init(model_float,
                                                                                                  self.representative_data_gen,
                                                                                                  fw_info=self.get_fw_info(),
                                                                                                  target_platform_capabilities=self.get_tpc())

        # PTQ model
        in_tensor = np.random.randn(1, *ptq_model.input_shape[1:])
        out_ptq_model = ptq_model(in_tensor)

        # QAT model
        qat_model = ptq_model
        if self.test_loading:
            qat_model.save('qat2model.h5')
            qat_model = mct.keras_load_quantized_model('qat2model.h5')
            os.remove('qat2model.h5')

        self.compare(qat_model,
                     finalize=False,
                     input_x=self.representative_data_gen(),
                     quantization_info=quantization_info)

        out_qat_model = qat_model(in_tensor)
        self.unit_test.assertTrue(
            np.isclose(np.linalg.norm(out_qat_model - out_ptq_model) / np.linalg.norm(out_ptq_model), 0, atol=1e-6))

        if self.finalize:
            # QAT finalize model
            qat_finalize_model = mct.keras_quantization_aware_training_finalize(qat_model)
            self.compare(qat_finalize_model,
                         finalize=True,
                         input_x=self.representative_data_gen(),
                         quantization_info=quantization_info)
            out_qat_finalize_model = qat_finalize_model(in_tensor)
            self.unit_test.assertTrue(
                np.isclose(np.linalg.norm(out_qat_finalize_model - out_ptq_model) / np.linalg.norm(out_ptq_model), 0,
                           atol=1e-6))

    def compare(self, qat_model, finalize=False, input_x=None, quantization_info=None):

        for layer in qat_model.layers:
            if isinstance(layer, qi.KerasQuantizationWrapper):
                # Check Activation quantizers
                if layer.is_activation_quantization:
                    for quantizer in layer.activation_quantizers:
                        if finalize:
                            self.unit_test.assertTrue(isinstance(quantizer, qi.BaseKerasInferableQuantizer))
                            q = QUANTIZATION_METHOD_2_ACTIVATION_QUANTIZER[self.activation_quantization_method]
                            self.unit_test.assertTrue(isinstance(layer.activation_quantizers[0], q))
                        else:
                            self.unit_test.assertTrue(isinstance(quantizer, qi.BaseKerasTrainableQuantizer))
                            q = METHOD2ACTQUANTIZER[mct.TrainingMethod.STE][self.activation_quantization_method]
                            self.unit_test.assertTrue(isinstance(layer.activation_quantizers[0], q))


                # Check Weight quantizers
                if layer.is_weights_quantization:
                    for name, quantizer in layer.weights_quantizers.items():
                        if finalize:
                            self.unit_test.assertTrue(isinstance(quantizer, qi.BaseKerasInferableQuantizer))
                            q = QUANTIZATION_METHOD_2_WEIGHTS_QUANTIZER[self.weights_quantization_method]
                            self.unit_test.assertTrue(isinstance(layer.weights_quantizers[KERNEL], q))
                        else:
                            self.unit_test.assertTrue(isinstance(quantizer, qi.BaseKerasTrainableQuantizer))
                            q = METHOD2WEIGHTQUANTIZER[mct.TrainingMethod.STE][self.weights_quantization_method]
                            self.unit_test.assertTrue(isinstance(layer.weights_quantizers[KERNEL], q))


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

import tensorflow as tf
import numpy as np
from mct_quantizers.common.get_all_subclasses import get_all_subclasses
from mct_quantizers.keras.quantizers import BaseKerasInferableQuantizer

from model_compression_toolkit.core import MixedPrecisionQuantizationConfigV2
from model_compression_toolkit.qat import TrainingMethod
from model_compression_toolkit.qat.keras.quantizer.base_keras_qat_quantizer import BaseKerasQATTrainableQuantizer
from model_compression_toolkit.trainable_infrastructure import KerasTrainableQuantizationWrapper
from mct_quantizers import QuantizationTarget, KerasActivationQuantizationHolder, KerasQuantizationWrapper
from mct_quantizers.common.base_inferable_quantizer import QuantizerID

from model_compression_toolkit.trainable_infrastructure import BaseKerasTrainableQuantizer
from model_compression_toolkit.trainable_infrastructure.common.base_trainable_quantizer import BaseTrainableQuantizer
from model_compression_toolkit.trainable_infrastructure.keras.load_model import \
    keras_load_quantized_model
from tests.keras_tests.feature_networks_tests.feature_networks.mixed_precision_tests import \
    MixedPrecisionActivationBaseTest
from tests.keras_tests.tpc_keras import get_tpc
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
import model_compression_toolkit as mct

import os
from model_compression_toolkit.core.keras.default_framework_info import KERNEL
from tests.keras_tests.utils import get_layers_from_model_by_type

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

    def run_test(self, **kwargs):
        model_float = self.create_networks()
        ptq_model, quantization_info, custom_objects = mct.qat.keras_quantization_aware_training_init(model_float,
                                                                                                  self.representative_data_gen,
                                                                                                  fw_info=self.get_fw_info(),
                                                                                                  target_platform_capabilities=self.get_tpc())

        ptq_model2 = None
        if self.test_loading:
            ptq_model.save('qat2model.h5')
            ptq_model2 = keras_load_quantized_model('qat2model.h5')
            os.remove('qat2model.h5')

        if self.finalize:
            ptq_model = mct.qat.keras_quantization_aware_training_finalize(ptq_model)

        self.compare(ptq_model,
                     model_float,
                     ptq_model2,
                     input_x=self.representative_data_gen(),
                     quantization_info=quantization_info)

    def compare(self, quantized_model, float_model, loaded_model, input_x=None, quantization_info=None):
        if self.test_loading:
            for lo, ll in zip(quantized_model.layers, loaded_model.layers):
                if isinstance(ll, KerasTrainableQuantizationWrapper):
                    self.unit_test.assertTrue(isinstance(lo, KerasTrainableQuantizationWrapper))
                if isinstance(lo, KerasTrainableQuantizationWrapper):
                    self.unit_test.assertTrue(isinstance(ll, KerasTrainableQuantizationWrapper))
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
        dw_layers = get_layers_from_model_by_type(quantized_model, layers.DepthwiseConv2D)
        if self.finalize:
            self.unit_test.assertTrue(len(dw_layers)==1)
            dw_weight = float_model.layers[1].weights[0].numpy()
            quantized_dw_weight = dw_layers[0].weights[0].numpy()
        else:
            holder_layers = get_layers_from_model_by_type(quantized_model, KerasActivationQuantizationHolder)
            self.unit_test.assertTrue(len(holder_layers)==2, f'Expected to have 2 activation quantizers (for input and linear layers) but found {len(holder_layers)}')
            for name, quantizer in dw_layers[0].weights_quantizers.items():
                w_select = [w for w in float_model.layers[1].weights if name + ":0" in w.name]
                if len(w_select) != 1:
                    raise Exception()
                dw_weight = w_select[0]
                quantized_dw_weight = quantizer(dw_weight, False)
        self.unit_test.assertTrue(np.all(dw_weight == quantized_dw_weight))


class QuantizationAwareTrainingQuantizerHolderTest(QuantizationAwareTrainingTest):

    def __init__(self, unit_test):
        super().__init__(unit_test,
                         layer=layers.Conv2D(3,3))

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        outputs = self.layer(inputs)
        return keras.Model(inputs=inputs, outputs=outputs)

    def compare(self, quantized_model, float_model, loaded_model, input_x=None, quantization_info=None):
        holder_layers = get_layers_from_model_by_type(quantized_model, KerasActivationQuantizationHolder)
        self.unit_test.assertTrue(len(holder_layers)==2)
        for holder_layer in holder_layers:
            self.unit_test.assertTrue(holder_layer.get_config()['activation_holder_quantizer']['class_name'] == 'STEActivationQATQuantizer')
            self.unit_test.assertTrue(holder_layer.get_config()['activation_holder_quantizer']['config']['quantization_config']['activation_n_bits'] == 4)

            # Assert weights of quantizer were added to the quantization holder layer
            w_names = [w.name for w in holder_layer.weights]
            self.unit_test.assertTrue(any(['out_ptq_threshold_tensor' in w for w in w_names]))
            self.unit_test.assertTrue(any(['out_min' in w for w in w_names]))
            self.unit_test.assertTrue(any(['out_max' in w for w in w_names]))


class QATWrappersTest(BaseKerasFeatureNetworkTest):
    def __init__(self, unit_test, layer, weight_bits=2, activation_bits=4, finalize=True,
                 weights_quantization_method=mct.target_platform.QuantizationMethod.POWER_OF_TWO,
                 activation_quantization_method=mct.target_platform.QuantizationMethod.POWER_OF_TWO,
                 per_channel=True,
                 test_loading=False):
        self.layer = layer
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.finalize = finalize
        self.weights_quantization_method = weights_quantization_method
        self.activation_quantization_method = activation_quantization_method
        self.per_channel = per_channel
        self.test_loading = test_loading
        super().__init__(unit_test)

    def get_tpc(self):
        return get_tpc("QAT_wrappers_test", weight_bits=self.weight_bits, activation_bits=self.activation_bits,
                       weights_quantization_method=self.weights_quantization_method,
                       activation_quantization_method=self.activation_quantization_method,
                       per_channel=self.per_channel)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        outputs = self.layer(inputs)
        return keras.Model(inputs=inputs, outputs=outputs)

    def run_test(self, **kwargs):
        model_float = self.create_networks()
        ptq_model, quantization_info, custom_objects = mct.qat.keras_quantization_aware_training_init(model_float,
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
            qat_model = keras_load_quantized_model('qat2model.h5')
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
            qat_finalize_model = mct.qat.keras_quantization_aware_training_finalize(qat_model)
            for layer in qat_finalize_model.layers:
                if isinstance(layer, KerasQuantizationWrapper):
                    for q in layer.weights_quantizers.values():
                        self.unit_test.assertFalse(isinstance(q, BaseTrainableQuantizer), f'Quantizer {type(q)} is trainable after finalizing the model')
                elif isinstance(layer, KerasActivationQuantizationHolder):
                    q = layer.activation_holder_quantizer
                    self.unit_test.assertFalse(isinstance(q, BaseTrainableQuantizer), f'Quantizer {type(q)} is trainable after finalizing the model')

            self.compare(qat_finalize_model,
                         finalize=True,
                         input_x=self.representative_data_gen(),
                         quantization_info=quantization_info)
            out_qat_finalize_model = qat_finalize_model(in_tensor)
            self.unit_test.assertTrue(
                np.isclose(np.linalg.norm(out_qat_finalize_model - out_ptq_model) / np.linalg.norm(out_ptq_model), 0,
                           atol=1e-6))

    def compare(self, qat_model, finalize=False, input_x=None, quantization_info=None):
        all_trainable_quantizers = get_all_subclasses(BaseKerasQATTrainableQuantizer)
        all_inferable_quantizers = get_all_subclasses(BaseKerasInferableQuantizer)
        for layer in qat_model.layers:
            if isinstance(layer, KerasActivationQuantizationHolder):
                # Check Activation quantizers
                if finalize:
                    self.unit_test.assertTrue(isinstance(layer.activation_holder_quantizer, BaseKerasInferableQuantizer))
                    q = [_q for _q in all_inferable_quantizers if
                         _q.quantization_target == QuantizationTarget.Activation
                         and self.activation_quantization_method in _q.quantization_method
                         and _q.identifier == QuantizerID.INFERABLE]
                    self.unit_test.assertTrue(len(q) == 1)
                    self.unit_test.assertTrue(isinstance(layer.activation_holder_quantizer, q[0]))
                else:
                    self.unit_test.assertTrue(isinstance(layer.activation_holder_quantizer, BaseKerasTrainableQuantizer))
                    q = [_q for _q in all_trainable_quantizers if
                         _q.identifier == mct.qat.TrainingMethod.STE
                         and _q.quantization_target == QuantizationTarget.Activation
                         and self.activation_quantization_method in _q.quantization_method
                         and type(_q.identifier) == TrainingMethod]
                    self.unit_test.assertTrue(len(q) == 1)
                    self.unit_test.assertTrue(isinstance(layer.activation_holder_quantizer, q[0]))

            elif isinstance(layer, KerasQuantizationWrapper):
                # Check Weight quantizers
                if layer.is_weights_quantization:
                    for name, quantizer in layer.weights_quantizers.items():
                        if finalize:
                            self.unit_test.assertTrue(isinstance(quantizer, BaseKerasInferableQuantizer))
                            q = [_q for _q in all_inferable_quantizers if
                                 _q.quantization_target == QuantizationTarget.Weights
                                 and self.weights_quantization_method in _q.quantization_method
                                 and _q.identifier == QuantizerID.INFERABLE]
                            self.unit_test.assertTrue(len(q) == 1)
                            self.unit_test.assertTrue(isinstance(layer.weights_quantizers[KERNEL], q[0]))
                        else:
                            self.unit_test.assertTrue(isinstance(quantizer, BaseKerasTrainableQuantizer))
                            q = [_q for _q in all_trainable_quantizers if _q.identifier == mct.qat.TrainingMethod.STE
                                 and _q.quantization_target == QuantizationTarget.Weights
                                 and self.weights_quantization_method in _q.quantization_method
                                 and type(_q.identifier) == TrainingMethod]
                            self.unit_test.assertTrue(len(q) == 1)
                            self.unit_test.assertTrue(isinstance(layer.weights_quantizers[KERNEL], q[0]))


class QATWrappersMixedPrecisionCfgTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test, kpi_weights=np.inf, kpi_activation=np.inf, expected_mp_cfg=[0, 0, 0, 0]):
        self.kpi_weights = kpi_weights
        self.kpi_activation = kpi_activation
        self.expected_mp_cfg = expected_mp_cfg
        super().__init__(unit_test, activation_layers_idx=[1, 3, 6])

    def run_test(self, **kwargs):
        model_float = self.create_networks()
        config = mct.core.CoreConfig(mixed_precision_config=MixedPrecisionQuantizationConfigV2())
        qat_ready_model, quantization_info, custom_objects = mct.qat.keras_quantization_aware_training_init(
            model_float,
            self.representative_data_gen_experimental,
            mct.core.KPI(weights_memory=self.kpi_weights, activation_memory=self.kpi_activation),
            core_config=config,
            fw_info=self.get_fw_info(),
            target_platform_capabilities=self.get_tpc())

        self.compare(qat_ready_model, quantization_info)

    def compare(self, qat_ready_model, quantization_info):

        # check that MP search returns 8 bits configuration for all layers
        self.unit_test.assertTrue(all(quantization_info.mixed_precision_cfg == self.expected_mp_cfg))

        # check that quantizer gets multiple bits configuration
        for layer in qat_ready_model.layers:
            if isinstance(layer, KerasQuantizationWrapper):
                if layer.is_weights_quantization:
                    self.unit_test.assertTrue(
                        len(layer.weights_quantizers['kernel'].quantization_config.weights_bits_candidates) > 1)
            elif isinstance(layer, KerasActivationQuantizationHolder):
                self.unit_test.assertTrue(
                    len(layer.activation_holder_quantizer.quantization_config.activation_bits_candidates) > 1)

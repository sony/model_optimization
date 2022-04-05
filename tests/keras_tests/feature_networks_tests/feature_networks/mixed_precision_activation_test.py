# Copyright 2022 Sony Semiconductors Israel, Inc. All rights reserved.
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


import numpy as np
import tensorflow as tf

from model_compression_toolkit.hardware_models.default_hwm import generate_hardware_model, get_op_quantization_configs
from model_compression_toolkit.hardware_models.keras_hardware_model.keras_default import generate_fhw_model_keras
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
from keras import backend as K

import model_compression_toolkit as mct
from model_compression_toolkit.common.mixed_precision.kpi import KPI
from model_compression_toolkit.common.mixed_precision.mixed_precision_quantization_config import \
    MixedPrecisionQuantizationConfig
from model_compression_toolkit.common.user_info import UserInformation
from tests.common_tests.helpers.tensors_compare import cosine_similarity

keras = tf.keras
layers = keras.layers
hw_model = mct.hardware_representation


class MixedPrecisionActivationBaseTest(BaseKerasFeatureNetworkTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_fw_hw_model(self):
        eight_bits = hw_model.OpQuantizationConfig(
            activation_quantization_method=hw_model.QuantizationMethod.POWER_OF_TWO,
            weights_quantization_method=hw_model.QuantizationMethod.POWER_OF_TWO,
            activation_n_bits=8,
            weights_n_bits=8,
            weights_per_channel_threshold=True,
            enable_weights_quantization=True,
            enable_activation_quantization=True,
            quantization_preserving=False,
            fixed_scale=None,
            fixed_zero_point=None,
            weights_multiplier_nbits=None
        )

        weights_eight_bits = [eight_bits,
                              eight_bits.clone_and_edit(activation_n_bits=4),
                              eight_bits.clone_and_edit(activation_n_bits=2)]
        weights_four_bits = [eight_bits.clone_and_edit(weights_n_bits=4),
                             eight_bits.clone_and_edit(weights_n_bits=4, activation_n_bits=4),
                             eight_bits.clone_and_edit(weights_n_bits=4, activation_n_bits=2)]
        weights_two_bits = [eight_bits.clone_and_edit(weights_n_bits=2),
                            eight_bits.clone_and_edit(weights_n_bits=2, activation_n_bits=4),
                            eight_bits.clone_and_edit(weights_n_bits=2, activation_n_bits=2)]

        mixed_precision_cfg = weights_two_bits + weights_eight_bits + weights_four_bits

        base_config, mixed_precision_cfg_list = get_op_quantization_configs()
        hwm = generate_hardware_model(base_config, mixed_precision_cfg_list, name='mp_default_hwm')
        return generate_fhw_model_keras(name="mixed_precision_activation_test", hardware_model=hwm)

    def get_quantization_config(self):
        qc = mct.QuantizationConfig(mct.QuantizationErrorMethod.MSE,
                                    mct.QuantizationErrorMethod.MSE,
                                    relu_bound_to_power_of_2=False,
                                    weights_bias_correction=True,
                                    weights_per_channel_threshold=True,
                                    input_scaling=False,
                                    activation_channel_equalization=False)

        return MixedPrecisionQuantizationConfig(qc, num_of_images=1)

    def get_bit_widths_config(self):
        return None

    def get_input_shapes(self):
        return [[self.val_batch_size, 224, 244, 3]]

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(30, 40)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(50, 40)(x)
        outputs = layers.ReLU()(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model

    def compare(self, quantized_model, float_model, input_x=None, quantization_info: UserInformation = None):
        # This is a base test, so it does not check a thing. Only actual tests of mixed precision
        # compare things to test.
        raise NotImplementedError

    def get_split_candidates(self, mp_config, weights_layers_idx, activation_layers_idx, model_layers):
        fw_hw_model = self.get_fw_hw_model()
        layer2qco = fw_hw_model.layer2qco
        candidates = [[(qc.weights_n_bits, qc.activation_n_bits) for qc in layer2qco.get(type(layer)).quantization_config_list]
                      for layer in model_layers]
        for layer_candidates in candidates:
            layer_candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)

        weights_bits = [list(map(lambda c: c[0], candidates[i])) for i in np.array(mp_config)[weights_layers_idx]]
        activation_bits = [list(map(lambda c: c[1], candidates[i])) for i in np.array(mp_config)[activation_layers_idx]]

        return weights_bits, activation_bits


class MixedPrecisionActivationSearchTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_kpi(self):
        # kpi is infinity -> should give best model - 8bits on all layers for both weights and activations
        return KPI(np.inf, np.inf)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        weights_bits, activation_bits = self.get_split_candidates(mp_config=quantization_info.mixed_precision_cfg,
                                                                  weights_layers_idx=[1, 2],
                                                                  activation_layers_idx=[0, 1, 3],
                                                                  model_layers=float_model.layers)

        # kpi is infinity -> should give best model - 8bits
        # only layers 0, 1, 3 in the test model have activations that need to be quantized
        self.unit_test.assertTrue((activation_bits == [8, 8, 8]))
        # only layers 1, 2 in the test model have weights that need to be quantized
        self.unit_test.assertTrue((weights_bits == [8, 8]))

        # verify weights quantization
        for i in range(30):  # quantized per channel
            self.unit_test.assertTrue(
                np.unique(quantized_model.layers[2].weights[0][:, :, :, i]).flatten().shape[0] <= 256)
        for i in range(50):  # quantized per channel
            self.unit_test.assertTrue(
                np.unique(quantized_model.layers[4].weights[0][:, :, :, i]).flatten().shape[0] <= 256)

        # verify activation quantization
        inp = quantized_model.input  # input placeholder
        out = [layer.output for layer in quantized_model.layers]  # all layer outputs
        get_outputs = K.function([inp], out)
        layer_outs = get_outputs([input_x])
        # verifying fake quant nodes output
        self.unit_test.assertTrue(np.unique(layer_outs[1].flatten()).shape[0] <= 256)
        self.unit_test.assertTrue(np.unique(layer_outs[3].flatten()).shape[0] <= 256)
        self.unit_test.assertTrue(np.unique(layer_outs[6].flatten()).shape[0] <= 256)


class MixedPrecisionActivationSearchKPI4BitsAvgTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_kpi(self):
        # kpi is for 4 bits on average
        return KPI(weights_memory=2544000 * 4 / 8, activation_memory=2677486 * 4 / 8)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # only layers 0, 1, 3  in the test model have activations that need to be quantized
        # only layers 1, 2  layers in the test model have weights that need to be quantized
        weights_bits, activation_bits = self.get_split_candidates(mp_config=quantization_info.mixed_precision_cfg,
                                                                  weights_layers_idx=[1, 2],
                                                                  activation_layers_idx=[0, 1, 2, 3],
                                                                  model_layers=quantized_model.layers)

        # verify that at least one layer is quantized with less than 8 bits
        self.unit_test.assertTrue(np.any(np.asarray([weights_bits + activation_bits]) != 8))


class MixedPrecisionActivationSearchKPI2BitsAvgTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_kpi(self):
        # kpi is for 2 bits on average
        return KPI(weights_memory=2544000 * 2 / 8, activation_memory=2677486 * 2 / 8)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        weights_bits, activation_bits = self.get_split_candidates(mp_config=quantization_info.mixed_precision_cfg,
                                                                  weights_layers_idx=[1, 2],
                                                                  activation_layers_idx=[0, 1, 2, 3],
                                                                  model_layers=quantized_model.layers)
        # kpi is minimal -> should give minimal model - 2bits for
        # only layers 0, 1, 3  in the test model have activations that need to be quantized
        self.unit_test.assertTrue((activation_bits == [2, 2, 2, 2]))
        # only layers 1, 2  layers in the test model have weights that need to be quantized
        self.unit_test.assertTrue((weights_bits == [2, 2]))

        for i in range(30):  # quantized per channel
            self.unit_test.assertTrue(
                np.unique(quantized_model.layers[2].weights[0][:, :, :, i]).flatten().shape[0] <= 4)
        for i in range(50):  # quantized per channel
            self.unit_test.assertTrue(
                np.unique(quantized_model.layers[4].weights[0][:, :, :, i]).flatten().shape[0] <= 4)

        # verify activation quantization
        inp = quantized_model.input  # input placeholder
        out = [layer.output for layer in quantized_model.layers]  # all layer outputs
        get_outputs = K.function([inp], out)
        layer_outs = get_outputs([input_x])
        # verifying fake quant nodes output
        self.unit_test.assertTrue(np.unique(layer_outs[1].flatten()).shape[0] <= 4)
        self.unit_test.assertTrue(np.unique(layer_outs[3].flatten()).shape[0] <= 4)
        self.unit_test.assertTrue(np.unique(layer_outs[6].flatten()).shape[0] <= 4)


class MixedPrecisionActivationDepthwiseTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_kpi(self):
        return KPI(np.inf, np.inf)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.DepthwiseConv2D(30)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        model = keras.Model(inputs=inputs, outputs=x)
        return model

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        weights_bits, activation_bits = self.get_split_candidates(mp_config=quantization_info.mixed_precision_cfg,
                                                                  weights_layers_idx=[1],
                                                                  activation_layers_idx=[0, 2],
                                                                  model_layers=quantized_model.layers)

        # kpi is infinity -> should give best model - 8bits
        # only first and third layers in the test model have activations that need to be quantized
        self.unit_test.assertTrue((activation_bits == [8, 8]))

        # only second layer in the test model have weights that need to be quantized
        self.unit_test.assertTrue((weights_bits == [8]))

        y = float_model.predict(input_x)
        y_hat = quantized_model.predict(input_x)
        cs = cosine_similarity(y, y_hat)
        # quantifying both weights and activation so similarity is approximately within error range of 1e-4
        self.unit_test.assertTrue(np.isclose(cs, 1, rtol=1e-4), msg=f'fail cosine similarity check:{cs}')


class MixedPrecisionActivationSplitLayerTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = tf.split(inputs, num_or_size_splits=2, axis=1)
        c0 = layers.Conv2D(30, 40)(x[0])
        c1 = layers.Conv2D(30, 40)(x[1])
        model = keras.Model(inputs=inputs, outputs=[c0, c1])
        return model

    def get_kpi(self):
        # kpi is infinity -> should give best model - 8bits on all layers for both weights and activations
        return KPI(np.inf, np.inf)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        weights_bits, activation_bits = self.get_split_candidates(mp_config=quantization_info.mixed_precision_cfg,
                                                                  weights_layers_idx=[1, 2],
                                                                  activation_layers_idx=[0, 1, 3],
                                                                  model_layers=quantized_model.layers)

        # kpi is infinity -> should give best model - 8bits
        # only layers 0, 1, 3 in the test model have activations that need to be quantized
        self.unit_test.assertTrue((activation_bits == [8, 8, 8]))
        # only layers 1, 2 in the test model have weights that need to be quantized
        self.unit_test.assertTrue((weights_bits == [8, 8]))

        # verify weights quantization
        for i in range(30):  # quantized per channel
            self.unit_test.assertTrue(
                np.unique(quantized_model.layers[2].weights[0][:, :, :, i]).flatten().shape[0] <= 256)
        for i in range(50):  # quantized per channel
            self.unit_test.assertTrue(
                np.unique(quantized_model.layers[4].weights[0][:, :, :, i]).flatten().shape[0] <= 256)

        # verify activation quantization
        inp = quantized_model.input  # input placeholder
        out = [layer.output for layer in quantized_model.layers]  # all layer outputs
        get_outputs = K.function([inp], out)
        layer_outs = get_outputs([input_x])
        # verifying fake quant nodes output
        self.unit_test.assertTrue(np.unique(layer_outs[1].flatten()).shape[0] <= 256)
        self.unit_test.assertTrue(np.unique(layer_outs[3].flatten()).shape[0] <= 256)
        self.unit_test.assertTrue(np.unique(layer_outs[6].flatten()).shape[0] <= 256)

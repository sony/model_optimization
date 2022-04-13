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
from keras.layers import TFOpLambda

from tests.common_tests.helpers.activation_mp_hw_model import generate_hw_model_with_activation_mp, \
    generate_activation_mp_fhw_model_keras
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


def get_base_eight_bits_config_op():
    return hw_model.OpQuantizationConfig(
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


def get_base_mp_nbits_candidates():
    return [(4, 8), (4, 4), (4, 2),
            (8, 8), (8, 4), (8, 2),
            (2, 8), (2, 4), (2, 2)]


class MixedPrecisionActivationBaseTest(BaseKerasFeatureNetworkTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_fw_hw_model(self):
        eight_bits = get_base_eight_bits_config_op()

        # sets all combinations of 2, 4, 8 bits for weights and activations
        mixed_precision_candidates_list = get_base_mp_nbits_candidates()

        hwm = generate_hw_model_with_activation_mp(eight_bits, mixed_precision_candidates_list, name='mp_default_hwm')
        return generate_activation_mp_fhw_model_keras(name="mixed_precision_activation_test", hardware_model=hwm)

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
        layers_to_quantize = [layer for layer in model_layers
                              if (type(layer) not in fw_hw_model.get_layers_by_opset_name("NoQuantization")
                                  and type(layer) is not TFOpLambda)]
        layer2qco = fw_hw_model.layer2qco

        # get sorted candidates of each layer
        activation_layers_candidates = np.array(layers_to_quantize)[activation_layers_idx]
        weights_layers_candidates = np.array(layers_to_quantize)[weights_layers_idx]
        activation_candidates = [[(qc.weights_n_bits, qc.activation_n_bits) for qc in
                                 layer2qco.get(type(layer)).quantization_config_list] for layer in
                                 activation_layers_candidates]
        weights_candidates = [[(qc.weights_n_bits, qc.activation_n_bits) for qc in
                              layer2qco.get(type(layer)).quantization_config_list] for layer in
                              weights_layers_candidates]

        for layer_candidates in activation_candidates:
            layer_candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)

        for layer_candidates in weights_candidates:
            layer_candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)

        # get chosen n_bits for each layer (weights and activation separately)
        # NOTE: we assume that the order of the layers in the configuration is the same as it appears in model.layers,
        #   if this not the case, then this helper test function isn't valid.
        activation_bits = [activation_candidates[i][bitwidth_idx][1] for i, bitwidth_idx in
                           enumerate(np.array(mp_config)[activation_layers_idx])]
        weights_bits = [weights_candidates[i][bitwidth_idx][0] for i, bitwidth_idx in
                        enumerate(np.array(mp_config)[weights_layers_idx])]

        return weights_bits, activation_bits

    def verify_quantization(self, quantized_model, input_x, weights_layers_idx, weights_layers_channels_size,
                            activation_layers_idx, shape_limit):
        # verify weights quantization
        for i in range(len(weights_layers_idx)):
            for j in range(weights_layers_channels_size[i]):  # quantized per channel
                self.unit_test.assertTrue(
                    np.unique(quantized_model.layers[weights_layers_idx[i]].weights[0][:, :, :, j]).flatten().shape[
                        0] <= shape_limit)

        # verify activation quantization
        inp = quantized_model.input  # input placeholder
        out = [layer.output for layer in quantized_model.layers]  # all layer outputs
        get_outputs = K.function([inp], out)
        layer_outs = get_outputs([input_x])
        # verifying fake quant nodes output
        for idx in activation_layers_idx:
            self.unit_test.assertTrue(np.unique(layer_outs[idx].flatten()).shape[0] <= shape_limit)


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

        self.verify_quantization(quantized_model, input_x,
                                 weights_layers_idx=[2, 4],
                                 weights_layers_channels_size=[30, 50],
                                 activation_layers_idx=[1, 3, 6],
                                 shape_limit=256)


class MixedPrecisionActivationSearchKPI4BitsAvgTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_kpi(self):
        # kpi is for 4 bits on average
        return KPI(weights_memory=2544000 * 4 / 8, activation_memory=3725318 * 4 / 8)

    def get_fw_hw_model(self):
        eight_bits = get_base_eight_bits_config_op()
        # set only 8 and 4 bit candidates for test, to verify that all layers get exactly 4 bits
        mixed_precision_candidates_list = [(8, 8), (8, 4), (4, 8), (4, 4)]

        hwm = generate_hw_model_with_activation_mp(eight_bits, mixed_precision_candidates_list, name='mp_default_hwm')
        return generate_activation_mp_fhw_model_keras(name="mixed_precision_4bit_test", hardware_model=hwm)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        # only layers 0, 1, 3  in the test model have activations that need to be quantized
        # only layers 1, 2  layers in the test model have weights that need to be quantized
        weights_bits, activation_bits = self.get_split_candidates(mp_config=quantization_info.mixed_precision_cfg,
                                                                  weights_layers_idx=[1, 2],
                                                                  activation_layers_idx=[0, 1, 3],
                                                                  model_layers=float_model.layers)

        # kpi is 4 bit average -> should quantize all layers with 4 bits
        # only layers 0, 1, 3  in the test model have activations that need to be quantized
        self.unit_test.assertTrue((activation_bits == [4, 4, 4]))
        # only layers 1, 2  layers in the test model have weights that need to be quantized
        self.unit_test.assertTrue((weights_bits == [4, 4]))


class MixedPrecisionActivationSearchKPI2BitsAvgTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_kpi(self):
        # kpi is for 2 bits on average
        return KPI(weights_memory=2544000.0 * 2 / 8, activation_memory=3725318.0 * 2 / 8)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        weights_bits, activation_bits = self.get_split_candidates(mp_config=quantization_info.mixed_precision_cfg,
                                                                  weights_layers_idx=[1, 2],
                                                                  activation_layers_idx=[0, 1, 3],
                                                                  model_layers=float_model.layers)
        # kpi is minimal -> should give minimal model - 2bits for
        # only layers 0, 1, 3  in the test model have activations that need to be quantized
        self.unit_test.assertTrue((activation_bits == [2, 2, 2]))
        # only layers 1, 2  layers in the test model have weights that need to be quantized
        self.unit_test.assertTrue((weights_bits == [2, 2]))

        self.verify_quantization(quantized_model, input_x,
                                 weights_layers_idx=[2, 4],
                                 weights_layers_channels_size=[30, 50],
                                 activation_layers_idx=[1, 3, 6],
                                 shape_limit=4)


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
                                                                  model_layers=float_model.layers)

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


class MixedPrecisionActivationDepthwise4BitTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_kpi(self):
        # return KPI(np.inf, np.inf)
        return KPI(2700.0 * 4 / 8, 415518.0 * 4 / 8)

    def get_fw_hw_model(self):
        eight_bits = get_base_eight_bits_config_op()
        # set only 8 and 4 bit candidates for test, to verify that all layers get exactly 4 bits
        mixed_precision_candidates_list = [(8, 8), (8, 4), (4, 8), (4, 4)]

        hwm = generate_hw_model_with_activation_mp(eight_bits, mixed_precision_candidates_list, name='mp_default_hwm')
        return generate_activation_mp_fhw_model_keras(name="mixed_precision_depthwise_4bit_test", hardware_model=hwm)

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
                                                                  model_layers=float_model.layers)

        # kpi is 4 bit average -> should quantize all layers with 4 bits
        # only first and third layers in the test model have activations that need to be quantized
        self.unit_test.assertTrue((activation_bits == [4, 4]))

        # only second layer in the test model have weights that need to be quantized
        self.unit_test.assertTrue((weights_bits == [4]))


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
                                                                  activation_layers_idx=[0, 1, 2],
                                                                  model_layers=float_model.layers)

        # kpi is infinity -> should give best model - 8bits
        # only layers 0, 1, 2 in the test model have activations that need to be quantized
        self.unit_test.assertTrue((activation_bits == [8, 8, 8]))
        # only layers 1, 2 in the test model have weights that need to be quantized
        self.unit_test.assertTrue((weights_bits == [8, 8]))

        self.verify_quantization(quantized_model, input_x,
                                 weights_layers_idx=[3, 4],
                                 weights_layers_channels_size=[30, 30],
                                 activation_layers_idx=[1, 5, 6],
                                 shape_limit=256)


class MixedPrecisionActivationOnlyTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(30, 40)(inputs)
        x = layers.BatchNormalization()(x)
        outputs = layers.Conv2D(30, 40)(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model

    def get_fw_hw_model(self):
        eight_bits = get_base_eight_bits_config_op()
        mixed_precision_candidates_list = [(8, 8), (8, 4), (8, 2)]

        hwm = generate_hw_model_with_activation_mp(eight_bits, mixed_precision_candidates_list, name='mp_default_hwm')
        return generate_activation_mp_fhw_model_keras(name="mixed_precision_activation_weights_disabled_test", hardware_model=hwm)

    def get_kpi(self):
        # kpi is infinity -> should give best model - 8bits on all layers for both weights and activations
        return KPI(np.inf, np.inf)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        weights_bits, activation_bits = self.get_split_candidates(mp_config=quantization_info.mixed_precision_cfg,
                                                                  weights_layers_idx=[],
                                                                  activation_layers_idx=[0, 1, 2],
                                                                  model_layers=float_model.layers)

        # kpi is infinity -> should give best model - 8bits
        # only layers 0, 1, 3 in the test model have activations that need to be quantized
        self.unit_test.assertTrue((activation_bits == [8, 8, 8]))

        self.verify_quantization(quantized_model, input_x,
                                 weights_layers_idx=[],
                                 weights_layers_channels_size=[],
                                 activation_layers_idx=[1, 3, 5],
                                 shape_limit=256)


class MixedPrecisionActivationOnlyWeightsDisabledTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(30, 40)(inputs)
        x = layers.BatchNormalization()(x)
        outputs = layers.Conv2D(30, 40)(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model

    def get_fw_hw_model(self):
        eight_bits = get_base_eight_bits_config_op()
        weights_disabled_config = eight_bits.clone_and_edit(enable_weights_quantization=False)

        mixed_precision_candidates_list = [(8, 8), (8, 4), (8, 2)]

        hwm = generate_hw_model_with_activation_mp(weights_disabled_config, mixed_precision_candidates_list, name='mp_default_hwm')
        return generate_activation_mp_fhw_model_keras(name="mixed_precision_activation_weights_disabled_test", hardware_model=hwm)

    def get_kpi(self):
        # kpi is infinity -> should give best model - 8bits on all layers for both weights and activations
        return KPI(np.inf, np.inf)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        weights_bits, activation_bits = self.get_split_candidates(mp_config=quantization_info.mixed_precision_cfg,
                                                                  weights_layers_idx=[],
                                                                  activation_layers_idx=[0, 1, 2],
                                                                  model_layers=float_model.layers)

        # kpi is infinity -> should give best model - 8bits
        # only layers 0, 1, 3 in the test model have activations that need to be quantized
        self.unit_test.assertTrue((activation_bits == [8, 8, 8]))

        self.verify_quantization(quantized_model, input_x,
                                 weights_layers_idx=[],
                                 weights_layers_channels_size=[],
                                 activation_layers_idx=[1, 3, 5],
                                 shape_limit=256)


class MixedPrecisionActivationAddLayerTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_kpi(self):
        return KPI(np.inf, np.inf)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(30, 40)(inputs)
        x = layers.Add()([x, x])
        model = keras.Model(inputs=inputs, outputs=x)
        return model

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        weights_bits, activation_bits = self.get_split_candidates(mp_config=quantization_info.mixed_precision_cfg,
                                                                  weights_layers_idx=[1],
                                                                  activation_layers_idx=[0, 1, 2],
                                                                  model_layers=float_model.layers)

        # kpi is infinity -> should give best model - 8bits
        self.unit_test.assertTrue((activation_bits == [8, 8, 8]))
        self.unit_test.assertTrue((weights_bits == [8]))

        self.verify_quantization(quantized_model, input_x,
                                 weights_layers_idx=[2],
                                 weights_layers_channels_size=[30],
                                 activation_layers_idx=[1, 3, 5],
                                 shape_limit=256)


class MixedPrecisionActivationMultipleInputsTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)
        self.num_of_inputs = 3

    def get_kpi(self):
        return KPI(np.inf, np.inf)

    def get_input_shapes(self):
        return [[self.val_batch_size, 224, 244, 3] for _ in range(self.num_of_inputs)]

    def create_networks(self):
        inputs_1 = layers.Input(shape=self.get_input_shapes()[0][1:])
        inputs_2 = layers.Input(shape=self.get_input_shapes()[0][1:])
        inputs_3 = layers.Input(shape=self.get_input_shapes()[0][1:])
        x1 = layers.Conv2D(30, 40)(inputs_1)
        x2 = layers.Conv2D(30, 40)(inputs_2)
        x3 = layers.Conv2D(30, 40)(inputs_3)
        outputs = layers.Concatenate()([x1, x2, x3])
        model = keras.Model(inputs=[inputs_1, inputs_2, inputs_3], outputs=outputs)
        return model

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        self.unit_test.assertTrue(len(quantized_model.inputs) == 3)
        weights_bits, activation_bits = self.get_split_candidates(mp_config=quantization_info.mixed_precision_cfg,
                                                                  weights_layers_idx=[],
                                                                  activation_layers_idx=[0, 1, 2, 3, 4, 5],
                                                                  model_layers=float_model.layers)

        # kpi is infinity -> should give best model - 8bits
        # validating just activation config since we can't extract the weights layers
        # (since the model is not sequential), but all layers have activations.
        self.unit_test.assertTrue((activation_bits == [8, 8, 8, 8, 8, 8]))


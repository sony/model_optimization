# Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
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

import model_compression_toolkit as mct
from mct_quantizers import QuantizationMethod, KerasQuantizationWrapper
from model_compression_toolkit import DefaultDict
from model_compression_toolkit.core.keras.constants import GAMMA, BETA
from model_compression_toolkit.target_platform_capabilities.target_platform import Signedness
from model_compression_toolkit.target_platform_capabilities.constants import KERNEL_ATTR, KERAS_KERNEL, BIAS, BIAS_ATTR
from tests.common_tests.helpers.generate_test_tp_model import generate_test_attr_configs, \
    DEFAULT_WEIGHT_ATTR_CONFIG, KERNEL_BASE_CONFIG, generate_test_op_qc, BIAS_CONFIG
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
from tests.keras_tests.utils import get_layers_from_model_by_type

keras = tf.keras
layers = keras.layers
tp = mct.target_platform


def _generate_bn_quantized_tpm(quantize_linear):
    attr_cfgs_dict = generate_test_attr_configs()

    default_attr_cfg = attr_cfgs_dict[DEFAULT_WEIGHT_ATTR_CONFIG]
    # We don't want other attributes besides the specific attributes that we define in the mapping to be quantized
    default_attr_cfg.clone_and_edit(enable_weights_quantization=False)

    kernel_cfg = attr_cfgs_dict[KERNEL_BASE_CONFIG]
    kernel_cfg.clone_and_edit(enable_weights_quantization=quantize_linear)

    # Enable BN attributes quantization with 8-bit, POT quantizer
    bn_attr_cfg = default_attr_cfg.clone_and_edit(enable_weights_quantization=True,
                                                  weights_quantization_method=QuantizationMethod.POWER_OF_TWO,
                                                  weights_n_bits=8,
                                                  weights_per_channel_threshold=False)

    linear_op_qc = generate_test_op_qc(default_weight_attr_config=default_attr_cfg,
                                       kernel_base_config=kernel_cfg,
                                       bias_config=attr_cfgs_dict[BIAS_CONFIG],
                                       enable_activation_quantization=False)

    bn_op_qc = tp.OpQuantizationConfig(enable_activation_quantization=False,
                                       default_weight_attr_config=default_attr_cfg,
                                       attr_weights_configs_mapping={BETA: bn_attr_cfg, GAMMA: bn_attr_cfg},
                                       activation_n_bits=8,
                                       supported_input_activation_n_bits=8,
                                       activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
                                       quantization_preserving=False,
                                       fixed_scale=None,
                                       fixed_zero_point=None,
                                       simd_size=32,
                                       signedness=Signedness.AUTO)

    default_op_qc = tp.OpQuantizationConfig(enable_activation_quantization=False,
                                            default_weight_attr_config=default_attr_cfg,
                                            attr_weights_configs_mapping={},
                                            activation_n_bits=8,
                                            supported_input_activation_n_bits=8,
                                            activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
                                            quantization_preserving=False,
                                            fixed_scale=None,
                                            fixed_zero_point=None,
                                            simd_size=32,
                                            signedness=Signedness.AUTO)

    default_configuration_options = tp.QuantizationConfigOptions([default_op_qc])
    linear_configuration_options = tp.QuantizationConfigOptions([linear_op_qc])
    bn_configuration_options = tp.QuantizationConfigOptions([bn_op_qc])

    generated_tpm = tp.TargetPlatformModel(default_configuration_options, name='bn_quantized_tpm')

    with generated_tpm:

        tp.OperatorsSet("Conv", linear_configuration_options)
        tp.OperatorsSet("BN", bn_configuration_options)

    return generated_tpm


def _generate_bn_quantized_tpc(tp_model):
    tpc = tp.TargetPlatformCapabilities(tp_model, name='bn_quantized_tpc')

    with tpc:
        tp.OperationsSetToLayers("Conv", [layers.Conv2D],
                                 attr_mapping={
                                     KERNEL_ATTR: DefaultDict(default_value=KERAS_KERNEL),
                                     BIAS_ATTR: DefaultDict(default_value=BIAS)})

        tp.OperationsSetToLayers("BN", [layers.BatchNormalization],
                                 attr_mapping={
                                     GAMMA: DefaultDict(default_value=GAMMA),
                                     BETA: DefaultDict(default_value=BETA)})

    return tpc


class BNAttributesQuantization(BaseKerasFeatureNetworkTest):

    def __init__(self, unit_test, quantize_linear, input_shape=(8, 8, 3)):
        super(BNAttributesQuantization, self).__init__(unit_test=unit_test, input_shape=input_shape)

        self.quantize_linear = quantize_linear

    def get_tpc(self):
        tpm = _generate_bn_quantized_tpm(self.quantize_linear)
        return _generate_bn_quantized_tpc(tpm)

    def get_quantization_config(self):
        return mct.core.QuantizationConfig(weights_bias_correction=True)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(128, 3)(inputs)
        # we add a relu BEFORE the BN layer to prevent it from folding, so we can test its quantization
        x = layers.ReLU()(x)
        x = layers.BatchNormalization(beta_initializer='glorot_uniform',
                                      gamma_initializer=tf.keras.initializers.RandomUniform(minval=0.0001, maxval=1.05),
                                      moving_mean_initializer='glorot_uniform',
                                      moving_variance_initializer=tf.keras.initializers.RandomUniform(minval=0.0001,
                                                                                                      maxval=1.05))(x)
        return tf.keras.models.Model(inputs=inputs, outputs=x)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        float_bn_layer = get_layers_from_model_by_type(float_model, layers.BatchNormalization, include_wrapped_layers=False)
        self.unit_test.assertTrue(len(float_bn_layer) == 1, "Expecting the float model to have exactly 1 BN layer")
        float_bn_layer = float_bn_layer[0]

        quant_bn_layer = get_layers_from_model_by_type(quantized_model, layers.BatchNormalization, include_wrapped_layers=True)
        self.unit_test.assertTrue(len(quant_bn_layer) == 1, "Expecting the quantized model to have exactly 1 BN layer")
        quant_bn_layer = quant_bn_layer[0]

        # Verify that the BN layer is wrapped with weights quantization wrapper
        self.unit_test.assertTrue(isinstance(quant_bn_layer, KerasQuantizationWrapper),
                                  "BN layer is supposed to be wrapped with a weights quantization wrapper")

        # Verify BN attributes quantization
        f_bn_weights = float_bn_layer.weights
        q_bn_weights = quant_bn_layer.get_quantized_weights()

        f_beta = [w for w in f_bn_weights if BETA in w.name]
        self.unit_test.assertTrue(len(f_beta) == 1, "Expecting float model BN layer to have a BETA attribute")
        f_beta = f_beta[0]
        q_beta = q_bn_weights.get(BETA)
        self.unit_test.assertTrue(q_beta is not None, "Expecting quantized model BN layer to have a BETA attribute")
        self.unit_test.assertTrue(np.any(f_beta != q_beta), "Float and quantized BETA attributes are expected to have different values")

        f_gamma = [w for w in f_bn_weights if GAMMA in w.name]
        self.unit_test.assertTrue(len(f_gamma) == 1, "Expecting float model BN layer to have a GAMMA attribute")
        f_gamma = f_gamma[0]
        q_gamma = q_bn_weights.get(GAMMA)
        self.unit_test.assertTrue(q_gamma is not None, "Expecting quantized model BN layer to have a GAMMA attribute")
        self.unit_test.assertTrue(np.any(f_gamma != q_gamma),
                                  "Float and quantized GAMMA attributes are expected to have different values")

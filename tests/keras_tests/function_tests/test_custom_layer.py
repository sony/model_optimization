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
import unittest

import numpy as np
import tensorflow as tf

import model_compression_toolkit as mct
import model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema as schema
from model_compression_toolkit.core import CoreConfig, QuantizationConfig
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import Signedness
from model_compression_toolkit.target_platform_capabilities.constants import BIAS_ATTR, KERNEL_ATTR
from model_compression_toolkit.target_platform_capabilities.target_platform import LayerFilterParams
from tests.common_tests.helpers.generate_test_tp_model import generate_test_attr_configs, DEFAULT_WEIGHT_ATTR_CONFIG, \
    KERNEL_BASE_CONFIG, BIAS_CONFIG

keras = tf.keras
layers = keras.layers


class CustomIdentity(keras.layers.Layer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, inputs):
        return inputs


class CustomIdentityWithArg(keras.layers.Layer):

    def __init__(self, dummy_arg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dummy_arg = dummy_arg

    def get_config(self):
        config = super().get_config()
        config.update({"dummy_arg": self.dummy_arg})
        return config

    def call(self, inputs):
        return inputs


def get_tpc():
    """
    Assuming a target hardware that uses power-of-2 thresholds and quantizes weights and activations
    to 2 and 3 bits, accordingly. Our assumed hardware does not require quantization of some layers
    (e.g. Flatten & Droupout).
    This function generates a TargetPlatformCapabilities with the above specification.

    Returns:
         TargetPlatformCapabilities object
    """
    tp = mct.target_platform
    attr_cfg = generate_test_attr_configs(kernel_lut_values_bitwidth=0)
    base_cfg = schema.OpQuantizationConfig(activation_quantization_method=tp.QuantizationMethod.POWER_OF_TWO,
                                           enable_activation_quantization=True,
                                           activation_n_bits=32,
                                           supported_input_activation_n_bits=32,
                                           default_weight_attr_config=attr_cfg[DEFAULT_WEIGHT_ATTR_CONFIG],
                                           attr_weights_configs_mapping={},
                                           quantization_preserving=False,
                                           fixed_scale=1.0,
                                           fixed_zero_point=0,
                                           simd_size=32,
                                           signedness=Signedness.AUTO)

    default_configuration_options = schema.QuantizationConfigOptions(quantization_configurations=tuple([base_cfg]))

    operator_set = [schema.OperatorsSet(name="NoQuantization",
                                        qc_options=default_configuration_options.clone_and_edit(
                                            enable_activation_quantization=False)
                                        .clone_and_edit_weight_attribute(enable_weights_quantization=False))]
    tp_model = schema.TargetPlatformModel(default_qco=default_configuration_options,
                                          operator_set=tuple(operator_set),
                                          tpc_minor_version=None,
                                          tpc_patch_version=None,
                                          tpc_platform_type=None,
                                          add_metadata=False)

    return tp_model


class TestCustomLayer(unittest.TestCase):

    def test_custom_layer_in_tpc(self):
        inputs = layers.Input(shape=(3, 3, 3))
        x = CustomIdentity()(inputs)
        x = CustomIdentityWithArg(0)(x)
        model = keras.Model(inputs=inputs, outputs=x)

        core_cfg = CoreConfig(quantization_config=QuantizationConfig(
            custom_tpc_opset_to_layer={"NoQuantization":
                                           ([CustomIdentity, LayerFilterParams(CustomIdentityWithArg, dummy_arg=0)],)}))

        q_model, _ = mct.ptq.keras_post_training_quantization(model,
                                                              lambda: [np.random.randn(1, 3, 3, 3)],
                                                              core_config=core_cfg,
                                                              target_platform_capabilities=get_tpc())

        # verify the custom layer is in the quantized model
        self.assertTrue(isinstance(q_model.layers[2], CustomIdentity), 'Custom layer should be in the quantized model')
        self.assertTrue(isinstance(q_model.layers[3], CustomIdentityWithArg),
                        'Custom layer should be in the quantized model')
        # verify the custom layer isn't quantized
        self.assertTrue(len(q_model.layers) == 4,
                        'Quantized model should have only 3 layers: Input, KerasActivationQuantizationHolder, CustomIdentity & CustomIdentityWithArg')

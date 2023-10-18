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

keras = tf.keras
layers = keras.layers


class CustomIdentity(keras.layers.Layer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
    default_config = tp.OpQuantizationConfig(
        activation_quantization_method=tp.QuantizationMethod.POWER_OF_TWO,
        weights_quantization_method=tp.QuantizationMethod.POWER_OF_TWO,
        activation_n_bits=32,
        weights_n_bits=8,
        weights_per_channel_threshold=True,
        enable_weights_quantization=True,
        enable_activation_quantization=True,
        quantization_preserving=False,
        fixed_scale=1.0,
        fixed_zero_point=0,
        weights_multiplier_nbits=0)

    default_configuration_options = tp.QuantizationConfigOptions([default_config])
    tp_model = tp.TargetPlatformModel(default_configuration_options)
    with tp_model:
        tp_model.set_quantization_format(quantization_format=tp.quantization_format.QuantizationFormat.FAKELY_QUANT)
        tp.OperatorsSet("NoQuantization",
                        tp.get_default_quantization_config_options().clone_and_edit(
                            enable_weights_quantization=False,
                            enable_activation_quantization=False))

    tpc = tp.TargetPlatformCapabilities(tp_model)
    with tpc:
        # No need to quantize Flatten and Dropout layers
        tp.OperationsSetToLayers("NoQuantization", [CustomIdentity])

    return tpc


class TestCustomLayer(unittest.TestCase):

    def test_custom_layer_in_tpc(self):
        inputs = layers.Input(shape=(3, 3, 3))
        x = CustomIdentity()(inputs)
        model = keras.Model(inputs=inputs, outputs=x)

        q_model, _ = mct.ptq.keras_post_training_quantization_experimental(model,
                                                                           lambda: [np.random.randn(1, 3, 3, 3)],
                                                                           target_platform_capabilities=get_tpc())

        # verify the custom layer is in the quantized model
        self.assertTrue(isinstance(q_model.layers[2], CustomIdentity), 'Custom layer should be in the quantized model')
        # verify the custom layer isn't quantized
        self.assertTrue(len(q_model.layers) == 3,
                        'Quantized model should have only 3 layers: Input, KerasActivationQuantizationHolder & CustomIdentity')

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

import numpy as np
import tensorflow as tf

from model_compression_toolkit.quantizers_infrastructure import KerasQuantizationWrapper
from model_compression_toolkit.quantizers_infrastructure.inferable_infrastructure.common.base_inferable_quantizer import \
    mark_quantizer, QuantizationTarget
from model_compression_toolkit.quantizers_infrastructure.inferable_infrastructure.keras.quantizers import \
    BaseKerasInferableQuantizer
from model_compression_toolkit.target_platform_capabilities.target_platform import QuantizationMethod
from tests.quantizers_infrastructure_tests.inferable_infrastructure_tests.base_inferable_quantizer_test import \
    BaseInferableQuantizerTest

keras = tf.keras
layers = keras.layers

WEIGHT = 'kernel'
CLASS_NAME = 'class_name'


@mark_quantizer(quantization_target=QuantizationTarget.Weights,
                quantization_method=[QuantizationMethod.POWER_OF_TWO, QuantizationMethod.SYMMETRIC])
class IdentityWeightsQuantizer(BaseKerasInferableQuantizer):
    """
    A dummy quantizer for test usage - "quantize" the layer's weights to the original weights
    """
    def __init__(self):
        super().__init__()

    def __call__(self,
                 inputs: tf.Tensor,
                 training: bool):
        return inputs

    def get_config(self):
        return {}


@mark_quantizer(quantization_target=QuantizationTarget.Weights,
                quantization_method=[QuantizationMethod.POWER_OF_TWO, QuantizationMethod.SYMMETRIC])
class ZeroWeightsQuantizer(BaseKerasInferableQuantizer):
    """
    A dummy quantizer for test usage - "quantize" the layer's weights to 0
    """
    def __init__(self):
        super().__init__()

    def __call__(self,
                 inputs: tf.Tensor,
                 training: bool):
        return inputs * 0

    def get_config(self):
        return {}


@mark_quantizer(quantization_target=QuantizationTarget.Activation,
                quantization_method=[QuantizationMethod.POWER_OF_TWO, QuantizationMethod.SYMMETRIC])
class ZeroActivationsQuantizer(BaseKerasInferableQuantizer):
    """
    A dummy quantizer for test usage - "quantize" the layer's activation to 0
    """
    def __init__(self):
        super().__init__()

    def __call__(self,
                 inputs: tf.Tensor,
                 training: bool = True) -> tf.Tensor:
        return inputs * 0


class TestKerasWeightsQuantizationWrapper(BaseInferableQuantizerTest):

    def __init__(self, unit_test):
        super().__init__(unit_test)

        self.input_shapes = [(1, 8, 8, 3)]
        self.inputs = [np.random.randn(*in_shape) for in_shape in self.input_shapes]

    def create_networks(self):
        inputs = layers.Input(shape=self.input_shapes[0][1:])
        x = layers.Conv2D(6, 7, use_bias=False)(inputs)
        return keras.Model(inputs=inputs, outputs=x)

    def run_test(self):
        model = self.create_networks()
        conv_layer = model.layers[1]

        wrapper = KerasQuantizationWrapper(conv_layer)
        wrapper.add_weights_quantizer(WEIGHT, IdentityWeightsQuantizer())

        # build
        wrapper.build(self.input_shapes)
        (name, weight, quantizer) = wrapper._weights_vars[0]
        self.unit_test.assertTrue(isinstance(wrapper, KerasQuantizationWrapper))
        self.unit_test.assertTrue(isinstance(wrapper.layer, layers.Conv2D))
        self.unit_test.assertTrue(name == WEIGHT)
        self.unit_test.assertTrue((weight == getattr(wrapper.layer, WEIGHT)).numpy().all())
        self.unit_test.assertTrue(isinstance(quantizer, IdentityWeightsQuantizer))

        # call
        call_inputs = self.inputs[0]
        outputs = wrapper.call(call_inputs.astype('float32'))
        self.unit_test.assertTrue((outputs == conv_layer(call_inputs)).numpy().all())


class TestKerasActivationsQuantizationWrapper(TestKerasWeightsQuantizationWrapper):

    def __init__(self, unit_test):
        super().__init__(unit_test)

    def run_test(self):
        model = self.create_networks()
        conv_layer = model.layers[1]

        wrapper = KerasQuantizationWrapper(conv_layer, activation_quantizers=[ZeroActivationsQuantizer()])

        # build
        wrapper.build(self.input_shapes)
        (act_quantizer) = wrapper._activation_vars[0]
        self.unit_test.assertTrue(isinstance(act_quantizer, ZeroActivationsQuantizer))

        # apply the wrapper on inputs
        call_inputs = self.inputs[0]
        outputs = wrapper.call(call_inputs.astype('float32'))
        self.unit_test.assertTrue((outputs == 0).numpy().all())

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
import tensorflow as tf

from model_compression_toolkit.target_platform_capabilities.target_platform import QuantizationMethod
from model_compression_toolkit.quantizers_infrastructure import QuantizationTarget
from model_compression_toolkit.quantizers_infrastructure.inferable_infrastructure.keras.quantizers import \
    BaseKerasInferableQuantizer, WeightsPOTInferableQuantizer, WeightsSymmetricInferableQuantizer, \
    WeightsUniformInferableQuantizer, ActivationPOTInferableQuantizer, ActivationSymmetricInferableQuantizer, \
    ActivationUniformInferableQuantizer
from tests.quantizers_infrastructure_tests.inferable_infrastructure_tests.keras.inferable_keras.test_activation_inferable_quantizers import \
    TestKerasActivationsSymmetricInferableQuantizer, \
    TestKerasActivationsUnsignedSymmetricInferableQuantizer, TestKerasActivationsIllegalPOTInferableQuantizer, \
    TestKerasActivationsPOTInferableQuantizer, TestKerasActivationsUnsignedPOTInferableQuantizer, \
    TestKerasActivationsUniformInferableQuantizer, TestKerasActivationsUniformInferableZeroNotInRange
from tests.quantizers_infrastructure_tests.inferable_infrastructure_tests.keras.inferable_keras.test_activation_lut_inferable_quantizer import \
    TestKerasActivationPOTLUTQuantizerAssertions, TestKerasActivationPOTLUTQuantizer
from tests.quantizers_infrastructure_tests.inferable_infrastructure_tests.keras.inferable_keras.test_activation_quantizer_holder import TestActivationQuantizationHolderInference, TestActivationQuantizationHolderSaveAndLoad
from tests.quantizers_infrastructure_tests.inferable_infrastructure_tests.keras.inferable_keras.test_get_quantizers import \
    TestGetInferableQuantizer
from tests.quantizers_infrastructure_tests.inferable_infrastructure_tests.keras.inferable_keras.test_weights_inferable_quantizer import \
    TestKerasWeightsPOTInferableQuantizerRaise, \
    TestKerasWeightsPOTInferableSignedPerTensorQuantizer, TestKerasWeightsPOTInferableSignedPerChannelQuantizer, \
    TestKerasWeightsSymmetricInferableQuantizerRaise, TestKerasWeightsSymmetricInferableSignedPerTensorQuantizer, \
    TestKerasWeightsSymmetricInferableSignedPerChannelQuantizer, TestKerasWeightsUniformInferableQuantizerRaise, \
    TestKerasWeightsUniformInferableSignedPerTensorQuantizer, TestKerasWeightsUniformInferableSignedPerChannelQuantizer, \
    TestKerasWeightsUniformInferableZeroNotInRange
from tests.quantizers_infrastructure_tests.inferable_infrastructure_tests.keras.inferable_keras.test_weights_lut_inferable_quantizer import \
    TestKerasWeightsSymmetricLUTQuantizer, TestKerasWeightsPOTLUTQuantizer, \
    TestKerasWeightsSymmetricLUTQuantizerAssertions, TestKerasWeightsLUTPOTQuantizerAssertions
from tests.quantizers_infrastructure_tests.inferable_infrastructure_tests.keras.inferable_keras.test_keras_quantization_wrapper import \
    TestKerasWeightsQuantizationWrapper, TestKerasActivationsQuantizationWrapper

layers = tf.keras.layers


class KerasInferableInfrastructureTestRunner(unittest.TestCase):

    def test_weights_inferable_quantizers(self):
        TestKerasWeightsPOTInferableQuantizerRaise(self).run_test()
        TestKerasWeightsPOTInferableSignedPerTensorQuantizer(self).run_test()
        TestKerasWeightsPOTInferableSignedPerChannelQuantizer(self).run_test()
        TestKerasWeightsSymmetricInferableQuantizerRaise(self).run_test()
        TestKerasWeightsSymmetricInferableSignedPerTensorQuantizer(self).run_test()
        TestKerasWeightsSymmetricInferableSignedPerChannelQuantizer(self).run_test()
        TestKerasWeightsUniformInferableQuantizerRaise(self).run_test()
        TestKerasWeightsUniformInferableSignedPerTensorQuantizer(self).run_test()
        TestKerasWeightsUniformInferableSignedPerChannelQuantizer(self).run_test()
        TestKerasWeightsUniformInferableZeroNotInRange(self).run_test()

    def test_activation_inferable_quantizers(self):
        TestKerasActivationsSymmetricInferableQuantizer(self).run_test()
        TestKerasActivationsUnsignedSymmetricInferableQuantizer(self).run_test()
        TestKerasActivationsIllegalPOTInferableQuantizer(self).run_test()
        TestKerasActivationsPOTInferableQuantizer(self).run_test()
        TestKerasActivationsUnsignedPOTInferableQuantizer(self).run_test()
        TestKerasActivationsUniformInferableQuantizer(self).run_test()
        TestKerasActivationsUniformInferableZeroNotInRange(self).run_test()

    def test_weights_inferable_lut_quantizer(self):
        TestKerasWeightsSymmetricLUTQuantizerAssertions(self).run_test()
        TestKerasWeightsSymmetricLUTQuantizer(self).run_test()
        TestKerasWeightsLUTPOTQuantizerAssertions(self).run_test()
        TestKerasWeightsPOTLUTQuantizer(self).run_test()

    def test_activation_inferable_lut_quantizer(self):
        TestKerasActivationPOTLUTQuantizerAssertions(self).run_test()
        TestKerasActivationPOTLUTQuantizer(self).run_test()

    def test_get_quantizers(self):
        TestGetInferableQuantizer(self, quant_target=QuantizationTarget.Weights,
                                  quant_method=QuantizationMethod.POWER_OF_TWO,
                                  quantizer_base_class=BaseKerasInferableQuantizer,
                                  expected_quantizer_class=WeightsPOTInferableQuantizer).run_test()
        TestGetInferableQuantizer(self, quant_target=QuantizationTarget.Weights,
                                  quant_method=QuantizationMethod.SYMMETRIC,
                                  quantizer_base_class=BaseKerasInferableQuantizer,
                                  expected_quantizer_class=WeightsSymmetricInferableQuantizer).run_test()
        TestGetInferableQuantizer(self, quant_target=QuantizationTarget.Weights,
                                  quant_method=QuantizationMethod.UNIFORM,
                                  quantizer_base_class=BaseKerasInferableQuantizer,
                                  expected_quantizer_class=WeightsUniformInferableQuantizer).run_test()
        TestGetInferableQuantizer(self, quant_target=QuantizationTarget.Activation,
                                  quant_method=QuantizationMethod.POWER_OF_TWO,
                                  quantizer_base_class=BaseKerasInferableQuantizer,
                                  expected_quantizer_class=ActivationPOTInferableQuantizer).run_test()
        TestGetInferableQuantizer(self, quant_target=QuantizationTarget.Activation,
                                  quant_method=QuantizationMethod.SYMMETRIC,
                                  quantizer_base_class=BaseKerasInferableQuantizer,
                                  expected_quantizer_class=ActivationSymmetricInferableQuantizer).run_test()
        TestGetInferableQuantizer(self, quant_target=QuantizationTarget.Activation,
                                  quant_method=QuantizationMethod.UNIFORM,
                                  quantizer_base_class=BaseKerasInferableQuantizer,
                                  expected_quantizer_class=ActivationUniformInferableQuantizer).run_test()

    def test_layer_keras_infrastructure(self):
        TestKerasWeightsQuantizationWrapper(self).run_test()
        TestKerasActivationsQuantizationWrapper(self).run_test()

    def test_activation_quantization_holder(self):
        TestActivationQuantizationHolderInference(self).run_test()
        TestActivationQuantizationHolderSaveAndLoad(self).run_test()


if __name__ == '__main__':
    unittest.main()

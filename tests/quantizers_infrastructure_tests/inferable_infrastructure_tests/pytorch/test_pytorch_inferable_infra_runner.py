# Copyright 2021 Sony Semiconductor Israel, Inc. All rights reserved.
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

from model_compression_toolkit.target_platform_capabilities.target_platform import QuantizationMethod
from model_compression_toolkit.quantizers_infrastructure import QuantizationTarget
from model_compression_toolkit.quantizers_infrastructure.inferable_infrastructure.pytorch.quantizers import \
    BasePyTorchInferableQuantizer, WeightsPOTInferableQuantizer, WeightsSymmetricInferableQuantizer, \
    WeightsUniformInferableQuantizer, ActivationPOTInferableQuantizer, ActivationSymmetricInferableQuantizer, \
    ActivationUniformInferableQuantizer, WeightsLUTPOTInferableQuantizer, WeightsLUTSymmetricInferableQuantizer, \
    ActivationLutPOTInferableQuantizer
from tests.quantizers_infrastructure_tests.inferable_infrastructure_tests.pytorch.inferable_pytorch.test_activation_lut_inferable_quantizer import \
    TestPytorchActivationPOTLUTQuantizerAssertions, TestPytorchActivationPOTLUTQuantizer
from tests.quantizers_infrastructure_tests.inferable_infrastructure_tests.pytorch.inferable_pytorch.test_activations_inferable_quantizer import \
    TestPytorchActivationsSymmetricInferableQuantizer, \
    TestPytorchActivationsUnsignedSymmetricInferableQuantizer, TestPytorchActivationsPOTInferableQuantizerRaise, \
    TestPytorchActivationsSignedPOTInferableQuantizer, TestPytorchActivationsUnsignedPOTInferableQuantizer, \
    TestPytorchActivationsUniformInferableQuantizer, TestPytorchActivationsUniformInferableZeroNotInRange
from tests.quantizers_infrastructure_tests.inferable_infrastructure_tests.pytorch.inferable_pytorch.test_get_quantizers import \
    TestGetInferableQuantizer
from tests.quantizers_infrastructure_tests.inferable_infrastructure_tests.pytorch.inferable_pytorch.test_weights_inferable_quantizer import \
    TestPytorchWeightsSymmetricInferablePerTensorQuantizer, TestPytorchWeightsSymmetricInferablePerChannelQuantizer, \
    TestPytorchWeightsSymmetricInferableNoAxisQuantizer, TestPytorchWeightsPOTInferableQuantizerRaise, \
    TestPytorchWeightsPOTInferablePerChannelQuantizer, TestPytorchWeightsUniformInferablePerChannelQuantizer, \
    TestPytorchWeightsUniformInferablePerTensorQuantizer, TestPytorchWeightsUniformInferableQuantizerZeroNotInRange
from tests.quantizers_infrastructure_tests.inferable_infrastructure_tests.pytorch.inferable_pytorch.test_weights_lut_inferable_quantizer import \
    TestPytorchWeightsSymmetricLUTQuantizerAssertions, TestPytorchWeightsSymmetricLUTQuantizer, \
    TestPytorchWeightsLUTPOTQuantizerAssertions, TestPytorchWeightsPOTLUTQuantizer
from tests.quantizers_infrastructure_tests.inferable_infrastructure_tests.pytorch.inferable_pytorch.test_pytorch_quantization_wrapper import \
    TestPytorchWeightsQuantizationWrapper, TestPytorchActivationQuantizationWrapper


class PytorchInferableInfrastructureTestRunner(unittest.TestCase):

    def test_weights_inferable_quantizers(self):
        TestPytorchWeightsSymmetricInferablePerTensorQuantizer(self).run_test()
        TestPytorchWeightsSymmetricInferablePerChannelQuantizer(self).run_test()
        TestPytorchWeightsSymmetricInferableNoAxisQuantizer(self).run_test()
        TestPytorchWeightsPOTInferableQuantizerRaise(self).run_test()
        TestPytorchWeightsPOTInferablePerChannelQuantizer(self).run_test()
        TestPytorchWeightsUniformInferablePerChannelQuantizer(self).run_test()
        TestPytorchWeightsUniformInferablePerTensorQuantizer(self).run_test()
        TestPytorchWeightsUniformInferableQuantizerZeroNotInRange(self).run_test()

    def test_activation_inferable_quantizers(self):
        TestPytorchActivationsSymmetricInferableQuantizer(self).run_test()
        TestPytorchActivationsUnsignedSymmetricInferableQuantizer(self).run_test()
        TestPytorchActivationsPOTInferableQuantizerRaise(self).run_test()
        TestPytorchActivationsSignedPOTInferableQuantizer(self).run_test()
        TestPytorchActivationsUnsignedPOTInferableQuantizer(self).run_test()
        TestPytorchActivationsUniformInferableQuantizer(self).run_test()
        TestPytorchActivationsUniformInferableZeroNotInRange(self).run_test()

    def test_weights_inferable_lut_quantizer(self):
        TestPytorchWeightsSymmetricLUTQuantizerAssertions(self).run_test()
        TestPytorchWeightsSymmetricLUTQuantizer(self).run_test()
        TestPytorchWeightsLUTPOTQuantizerAssertions(self).run_test()
        TestPytorchWeightsPOTLUTQuantizer(self).run_test()

    def test_activation_inferable_lut_quantizer(self):
        TestPytorchActivationPOTLUTQuantizerAssertions(self).run_test()
        TestPytorchActivationPOTLUTQuantizer(self).run_test()

    def test_pytorch_get_quantizers(self):
        TestGetInferableQuantizer(self, quant_target=QuantizationTarget.Weights,
                                  quant_method=QuantizationMethod.POWER_OF_TWO,
                                  quantizer_base_class=BasePyTorchInferableQuantizer,
                                  expected_quantizer_class=WeightsPOTInferableQuantizer).run_test()
        TestGetInferableQuantizer(self, quant_target=QuantizationTarget.Weights,
                                  quant_method=QuantizationMethod.SYMMETRIC,
                                  quantizer_base_class=BasePyTorchInferableQuantizer,
                                  expected_quantizer_class=WeightsSymmetricInferableQuantizer).run_test()
        TestGetInferableQuantizer(self, quant_target=QuantizationTarget.Weights,
                                  quant_method=QuantizationMethod.UNIFORM,
                                  quantizer_base_class=BasePyTorchInferableQuantizer,
                                  expected_quantizer_class=WeightsUniformInferableQuantizer).run_test()
        TestGetInferableQuantizer(self, quant_target=QuantizationTarget.Weights,
                                  quant_method=QuantizationMethod.LUT_SYM_QUANTIZER,
                                  quantizer_base_class=BasePyTorchInferableQuantizer,
                                  expected_quantizer_class=WeightsLUTSymmetricInferableQuantizer).run_test()
        TestGetInferableQuantizer(self, quant_target=QuantizationTarget.Weights,
                                  quant_method=QuantizationMethod.LUT_POT_QUANTIZER,
                                  quantizer_base_class=BasePyTorchInferableQuantizer,
                                  expected_quantizer_class=WeightsLUTPOTInferableQuantizer).run_test()
        TestGetInferableQuantizer(self, quant_target=QuantizationTarget.Activation,
                                  quant_method=QuantizationMethod.POWER_OF_TWO,
                                  quantizer_base_class=BasePyTorchInferableQuantizer,
                                  expected_quantizer_class=ActivationPOTInferableQuantizer).run_test()
        TestGetInferableQuantizer(self, quant_target=QuantizationTarget.Activation,
                                  quant_method=QuantizationMethod.SYMMETRIC,
                                  quantizer_base_class=BasePyTorchInferableQuantizer,
                                  expected_quantizer_class=ActivationSymmetricInferableQuantizer).run_test()
        TestGetInferableQuantizer(self, quant_target=QuantizationTarget.Activation,
                                  quant_method=QuantizationMethod.UNIFORM,
                                  quantizer_base_class=BasePyTorchInferableQuantizer,
                                  expected_quantizer_class=ActivationUniformInferableQuantizer).run_test()
        TestGetInferableQuantizer(self, quant_target=QuantizationTarget.Activation,
                                  quant_method=QuantizationMethod.LUT_POT_QUANTIZER,
                                  quantizer_base_class=BasePyTorchInferableQuantizer,
                                  expected_quantizer_class=ActivationLutPOTInferableQuantizer).run_test()

    def test_layer_pytorch_infrastructre(self):
        TestPytorchWeightsQuantizationWrapper(self).run_test()
        TestPytorchActivationQuantizationWrapper(self).run_test()


if __name__ == '__main__':
    unittest.main()
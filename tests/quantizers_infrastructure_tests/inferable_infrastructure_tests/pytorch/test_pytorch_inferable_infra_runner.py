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

from model_compression_toolkit.core.common.target_platform import QuantizationMethod
from model_compression_toolkit.quantizers_infrastructure import QuantizationTarget
from model_compression_toolkit.quantizers_infrastructure.inferable_infrastructure.pytorch.quantizers import \
    BasePyTorchInferableQuantizer
from tests.quantizers_infrastructure_tests.inferable_infrastructure_tests.pytorch.inferable_pytorch.test_activations_inferable_quantizer import \
    TestActivationPOTQuantizer, TestActivationSymmetricQuantizer, TestActivationUniformQuantizer
from tests.quantizers_infrastructure_tests.inferable_infrastructure_tests.pytorch.inferable_pytorch.test_get_quantizers import \
    TestGetInferableQuantizer
from tests.quantizers_infrastructure_tests.inferable_infrastructure_tests.pytorch.inferable_pytorch.test_weights_inferable_quantizer import \
    TestWeightsPOTQuantizer, TestWeightsSymmetricQuantizer, TestWeightsUniformQuantizer


class PytorchInferableInfrastructureTestRunner(unittest.TestCase):

    def test_weights_inferable_quantizers(self):
        TestWeightsPOTQuantizer()
        TestWeightsSymmetricQuantizer()
        TestWeightsUniformQuantizer()

    def test_activation_inferable_quantizers(self):
        TestActivationPOTQuantizer()
        TestActivationSymmetricQuantizer()
        TestActivationUniformQuantizer()

    def test_pytorch_get_quantizers(self):
        TestGetInferableQuantizer(self, quant_target=QuantizationTarget.Weights,
                                  quant_method=QuantizationMethod.POWER_OF_TWO,
                                  quantizer_base_class=BasePyTorchInferableQuantizer)
        TestGetInferableQuantizer(self, quant_target=QuantizationTarget.Weights,
                                  quant_method=QuantizationMethod.SYMMETRIC,
                                  quantizer_base_class=BasePyTorchInferableQuantizer)
        TestGetInferableQuantizer(self, quant_target=QuantizationTarget.Weights,
                                  quant_method=QuantizationMethod.UNIFORM,
                                  quantizer_base_class=BasePyTorchInferableQuantizer)
        TestGetInferableQuantizer(self, quant_target=QuantizationTarget.Activation,
                                  quant_method=QuantizationMethod.POWER_OF_TWO,
                                  quantizer_base_class=BasePyTorchInferableQuantizer)
        TestGetInferableQuantizer(self, quant_target=QuantizationTarget.Activation,
                                  quant_method=QuantizationMethod.SYMMETRIC,
                                  quantizer_base_class=BasePyTorchInferableQuantizer)
        TestGetInferableQuantizer(self, quant_target=QuantizationTarget.Activation,
                                  quant_method=QuantizationMethod.UNIFORM,
                                  quantizer_base_class=BasePyTorchInferableQuantizer)


if __name__ == '__main__':
    unittest.main()
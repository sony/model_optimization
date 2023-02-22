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

from model_compression_toolkit.core.common.target_platform import QuantizationMethod
from model_compression_toolkit.quantizers_infrastructure import QuantizationTarget
from model_compression_toolkit.quantizers_infrastructure.inferable_infrastructure.keras.quantizers import \
    BaseKerasInferableQuantizer
from tests.quantizers_infrastructure_tests.inferable_infrastructure_tests.keras.inferable_keras.test_activation_inferable_quantizers import \
    TestKerasActivationsPOTQuantizer, TestKerasActivationsSymmetricQuantizer, TestKerasActivationsUniformQuantizer
from tests.quantizers_infrastructure_tests.inferable_infrastructure_tests.keras.inferable_keras.test_get_quantizers import \
    TestGetInferableQuantizer
from tests.quantizers_infrastructure_tests.inferable_infrastructure_tests.keras.inferable_keras.test_weights_inferable_quantizer import \
    TestKerasWeightsPOTQuantizer, TestKerasWeightsSymmetricQuantizer, TestKerasWeightsUniformQuantizer

layers = tf.keras.layers


class KerasInferableInfrastructureTest(unittest.TestCase):

    def test_weights_inferable_quantizers(self):
        TestKerasWeightsPOTQuantizer()
        TestKerasWeightsSymmetricQuantizer()
        TestKerasWeightsUniformQuantizer()

    def test_activation_inferable_quantizers(self):
        TestKerasActivationsPOTQuantizer()
        TestKerasActivationsSymmetricQuantizer()
        TestKerasActivationsUniformQuantizer()

    def test_get_quantizers(self):
        TestGetInferableQuantizer(self, quant_target=QuantizationTarget.Weights,
                                  quant_method=QuantizationMethod.POWER_OF_TWO,
                                  quantizer_base_class=BaseKerasInferableQuantizer)
        TestGetInferableQuantizer(self, quant_target=QuantizationTarget.Weights,
                                  quant_method=QuantizationMethod.SYMMETRIC,
                                  quantizer_base_class=BaseKerasInferableQuantizer)
        TestGetInferableQuantizer(self, quant_target=QuantizationTarget.Weights,
                                  quant_method=QuantizationMethod.UNIFORM,
                                  quantizer_base_class=BaseKerasInferableQuantizer)
        TestGetInferableQuantizer(self, quant_target=QuantizationTarget.Activation,
                                  quant_method=QuantizationMethod.POWER_OF_TWO,
                                  quantizer_base_class=BaseKerasInferableQuantizer)
        TestGetInferableQuantizer(self, quant_target=QuantizationTarget.Activation,
                                  quant_method=QuantizationMethod.SYMMETRIC,
                                  quantizer_base_class=BaseKerasInferableQuantizer)
        TestGetInferableQuantizer(self, quant_target=QuantizationTarget.Activation,
                                  quant_method=QuantizationMethod.UNIFORM,
                                  quantizer_base_class=BaseKerasInferableQuantizer)


if __name__ == '__main__':
    unittest.main()

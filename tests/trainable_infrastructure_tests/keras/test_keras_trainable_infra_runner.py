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

from model_compression_toolkit.qat import TrainingMethod
from model_compression_toolkit.target_platform_capabilities.target_platform import QuantizationMethod
from mct_quantizers import QuantizationTarget
from model_compression_toolkit.qat.keras.quantizer.ste_rounding.symmetric_ste import STEWeightQATQuantizer, \
    STEActivationQATQuantizer
from model_compression_toolkit.qat.keras.quantizer.ste_rounding.uniform_ste import STEUniformWeightQATQuantizer, \
    STEUniformActivationQATQuantizer
from model_compression_toolkit.trainable_infrastructure import BaseKerasTrainableQuantizer
from tests.trainable_infrastructure_tests.keras.trainable_keras.test_get_quantizers import \
    TestGetTrainableQuantizer
from tests.trainable_infrastructure_tests.keras.trainable_keras.test_keras_base_quantizer import TestKerasBaseWeightsQuantizer, \
    TestKerasBaseActivationsQuantizer, TestKerasQuantizerWithoutMarkDecorator


layers = tf.keras.layers


class KerasTrainableInfrastructureTestRunner(unittest.TestCase):

    def test_keras_base_quantizer(self):
        TestKerasBaseWeightsQuantizer(self).run_test()
        TestKerasBaseActivationsQuantizer(self).run_test()
        TestKerasQuantizerWithoutMarkDecorator(self).run_test()

    def test_get_quantizers(self):
        TestGetTrainableQuantizer(self, quant_target=QuantizationTarget.Weights,
                                  quant_method=QuantizationMethod.POWER_OF_TWO,
                                  quantizer_base_class=BaseKerasTrainableQuantizer,
                                  quantizer_id=TrainingMethod.STE,
                                  expected_quantizer_class=STEWeightQATQuantizer).run_test()
        TestGetTrainableQuantizer(self, quant_target=QuantizationTarget.Weights,
                                  quant_method=QuantizationMethod.SYMMETRIC,
                                  quantizer_base_class=BaseKerasTrainableQuantizer,
                                  quantizer_id=TrainingMethod.STE,
                                  expected_quantizer_class=STEWeightQATQuantizer).run_test()
        TestGetTrainableQuantizer(self, quant_target=QuantizationTarget.Weights,
                                  quant_method=QuantizationMethod.UNIFORM,
                                  quantizer_base_class=BaseKerasTrainableQuantizer,
                                  quantizer_id=TrainingMethod.STE,
                                  expected_quantizer_class=STEUniformWeightQATQuantizer).run_test()
        TestGetTrainableQuantizer(self, quant_target=QuantizationTarget.Activation,
                                  quant_method=QuantizationMethod.POWER_OF_TWO,
                                  quantizer_base_class=BaseKerasTrainableQuantizer,
                                  quantizer_id=TrainingMethod.STE,
                                  expected_quantizer_class=STEActivationQATQuantizer).run_test()
        TestGetTrainableQuantizer(self, quant_target=QuantizationTarget.Activation,
                                  quant_method=QuantizationMethod.SYMMETRIC,
                                  quantizer_base_class=BaseKerasTrainableQuantizer,
                                  quantizer_id=TrainingMethod.STE,
                                  expected_quantizer_class=STEActivationQATQuantizer).run_test()
        TestGetTrainableQuantizer(self, quant_target=QuantizationTarget.Activation,
                                  quant_method=QuantizationMethod.UNIFORM,
                                  quantizer_base_class=BaseKerasTrainableQuantizer,
                                  quantizer_id=TrainingMethod.STE,
                                  expected_quantizer_class=STEUniformActivationQATQuantizer).run_test()



if __name__ == '__main__':
    unittest.main()

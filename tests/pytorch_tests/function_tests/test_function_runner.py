# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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

from model_compression_toolkit.gptq import RoundingType
from model_compression_toolkit.target_platform_capabilities.target_platform import QuantizationMethod
from tests.pytorch_tests.function_tests.bn_info_collection_test import BNInfoCollectionTest, \
    Conv2D2BNInfoCollectionTest, Conv2DBNChainInfoCollectionTest, BNChainInfoCollectionTest, \
    BNLayerInfoCollectionTest, INP2BNInfoCollectionTest
from tests.pytorch_tests.function_tests.get_gptq_config_test import TestGetGPTQConfig
from tests.pytorch_tests.function_tests.resource_utilization_data_test import TestResourceUtilizationDataBasicAllBitwidth, \
    TestResourceUtilizationDataBasicPartialBitwidth, TestResourceUtilizationDataComplexPartialBitwidth, TestResourceUtilizationDataComplesAllBitwidth
from tests.pytorch_tests.function_tests.layer_fusing_test import LayerFusingTest1, LayerFusingTest2, LayerFusingTest3, \
    LayerFusingTest4
from tests.pytorch_tests.function_tests.set_device_test import SetDeviceTest
from tests.pytorch_tests.function_tests.set_layer_to_bitwidth_test import TestSetLayerToBitwidthWeights, \
    TestSetLayerToBitwidthActivation
from tests.pytorch_tests.function_tests.test_hessian_service import FetchActivationHessianTest, FetchWeightsHessianTest, \
    FetchHessianNotEnoughSamplesThrowTest, FetchHessianNotEnoughSamplesSmallBatchThrowTest, \
    FetchComputeBatchLargerThanReprBatchTest, FetchHessianRequiredZeroTest, FetchHessianMultipleNodesTest, \
    DoubleFetchHessianTest
from tests.pytorch_tests.function_tests.test_lut_activation_quanitzer_fake_quant import TestLUTQuantizerFakeQuantSigned, \
    TestLUTQuantizerFakeQuantUnsigned
from tests.pytorch_tests.function_tests.test_sensitivity_eval_non_supported_output import \
    TestSensitivityEvalWithArgmaxNode
from tests.pytorch_tests.function_tests.test_hessian_info_calculator import WeightsHessianTraceBasicModelTest, \
    WeightsHessianTraceAdvanceModelTest, \
    WeightsHessianTraceMultipleOutputsModelTest, WeightsHessianTraceReuseModelTest, \
    ActivationHessianTraceBasicModelTest, ActivationHessianTraceAdvanceModelTest, \
    ActivationHessianTraceMultipleOutputsModelTest, ActivationHessianTraceReuseModelTest, \
    ActivationHessianOutputExceptionTest, ActivationHessianTraceMultipleInputsModelTest


class FunctionTestRunner(unittest.TestCase):

    def test_conv2d_bn_info_collection(self):
        """
        This test checks the BatchNorm info collection with con2d -> bn model.
        Checks that the bn prior info has been "folded" into the conv node.
        """
        BNInfoCollectionTest(self).run_test()

    def test_conv2d_2bn_info_collection(self):
        """
        This test checks the BatchNorm info collection with 2 BN layers.
        Checks that the bn prior info of the first bn has been "folded" into the conv node.
        """
        Conv2D2BNInfoCollectionTest(self).run_test()

    def test_conv2d_bn_chain_info_collection(self):
        """
        This test checks the BatchNorm info collection with conv2d and a chain of 2 BN layers.
        Checks that the bn prior info of the first bn has been "folded" into the conv node.
        """
        Conv2DBNChainInfoCollectionTest(self).run_test()

    def test_bn_chain_info_collection(self):
        """
        This test checks the BatchNorm info collection with chain of 2 BN layers.
        """
        BNChainInfoCollectionTest(self).run_test()

    def test_layers_bn_info_collection(self):
        """
        This test checks the BatchNorm info collection with types of layers.
        """
        BNLayerInfoCollectionTest(self).run_test()

    def test_inp2bn_bn_info_collection(self):
        """
        This test checks the BatchNorm info collection with 2 parallel BN layer.
        """
        INP2BNInfoCollectionTest(self).run_test()

    def test_ru_data_basic_all(self):
        """
        This test checks the resource utilization data Pytorch API.
        """
        TestResourceUtilizationDataBasicAllBitwidth(self).run_test()

    def test_ru_data_basic_partial(self):
        """
        This test checks the resource utilization data Pytorch API.
        """
        TestResourceUtilizationDataBasicPartialBitwidth(self).run_test()

    def test_ru_data_complex_all(self):
        """
        This test checks the resource utilization data Pytorch API.
        """
        TestResourceUtilizationDataComplesAllBitwidth(self).run_test()

    def test_ru_data_complex_partial(self):
        """
        This test checks the resource utilization data Pytorch API.
        """
        TestResourceUtilizationDataComplexPartialBitwidth(self).run_test()

    def test_activation_hessian_trace(self):
        """
        This test checks the activation hessian trace approximation in Pytorch.
        """
        ActivationHessianTraceBasicModelTest(self).run_test()
        ActivationHessianTraceAdvanceModelTest(self).run_test()
        ActivationHessianTraceMultipleOutputsModelTest(self).run_test()
        ActivationHessianTraceReuseModelTest(self).run_test()
        ActivationHessianOutputExceptionTest(self).run_test()
        ActivationHessianTraceMultipleInputsModelTest(self).run_test()

    def test_weights_hessian_trace(self):
        """
        This test checks the weights hessian trace approximation in Pytorch.
        """
        WeightsHessianTraceBasicModelTest(self).run_test()
        WeightsHessianTraceAdvanceModelTest(self).run_test()
        WeightsHessianTraceMultipleOutputsModelTest(self).run_test()
        WeightsHessianTraceReuseModelTest(self).run_test()

    def test_hessian_service(self):
        """
        This test checks the Hessian service features with pytorch computation workflow.
        """
        FetchActivationHessianTest(self).run_test()
        FetchWeightsHessianTest(self).run_test()
        FetchHessianNotEnoughSamplesThrowTest(self).run_test()
        FetchHessianNotEnoughSamplesSmallBatchThrowTest(self).run_test()
        FetchComputeBatchLargerThanReprBatchTest(self).run_test()
        FetchHessianRequiredZeroTest(self).run_test()
        FetchHessianMultipleNodesTest(self).run_test()
        DoubleFetchHessianTest(self).run_test()

    def test_layer_fusing(self):
        """
        This test checks the Fusion mechanism in Pytorch.
        """
        LayerFusingTest1(self).run_test()
        LayerFusingTest2(self).run_test()
        LayerFusingTest3(self).run_test()
        LayerFusingTest4(self).run_test()

    def test_mixed_precision_set_bitwidth(self):
        """
        This test checks the functionality of setting a configurable layer's weights bit-width for mixed precision
        layer wrapper.
        """
        TestSetLayerToBitwidthWeights(self).run_test()
        TestSetLayerToBitwidthActivation(self).run_test()

    def test_sensitivity_eval_not_supported_output(self):
        """
        This test verifies failure on non-supported output nodes in mixed precision with Hessian-based scores.
        """
        TestSensitivityEvalWithArgmaxNode(self).run_test()

    def test_lut_activation_quantizer(self):
        """
        Test LUT activation Quantizer fakely quantized function.
        """
        TestLUTQuantizerFakeQuantSigned(self).run_test()
        TestLUTQuantizerFakeQuantUnsigned(self).run_test()

    def test_get_gptq_config(self):
        """
        This test checks the GPTQ config.
        """
        TestGetGPTQConfig(self).run_test()
        TestGetGPTQConfig(self, quantization_method=QuantizationMethod.POWER_OF_TWO).run_test()
        TestGetGPTQConfig(self, quantization_method=QuantizationMethod.POWER_OF_TWO).run_test()
        TestGetGPTQConfig(self, rounding_type=RoundingType.SoftQuantizer).run_test()
        TestGetGPTQConfig(self, quantization_method=QuantizationMethod.POWER_OF_TWO,
                          rounding_type=RoundingType.SoftQuantizer).run_test()
        TestGetGPTQConfig(self, quantization_method=QuantizationMethod.POWER_OF_TWO,
                          rounding_type=RoundingType.SoftQuantizer, train_bias=True).run_test()
        TestGetGPTQConfig(self, quantization_method=QuantizationMethod.POWER_OF_TWO,
                          rounding_type=RoundingType.SoftQuantizer, quantization_parameters_learning=True).run_test()

    def test_set_working_device(self):
        SetDeviceTest(self).run_test()


if __name__ == '__main__':
    unittest.main()

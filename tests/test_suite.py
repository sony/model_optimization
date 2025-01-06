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


#  ----------------- Unit test framework
import importlib
import unittest
from packaging import version

from tests.common_tests.function_tests.test_collectors_manipulation import TestCollectorsManipulations
from tests.common_tests.function_tests.test_edge_matcher import TestEdgeMatcher
#  ----------------  Individual test suites
from tests.common_tests.function_tests.test_histogram_collector import TestHistogramCollector
from tests.common_tests.function_tests.test_immutable_class import TestImmutableClass
from tests.common_tests.function_tests.test_logger import TestLogger
from tests.common_tests.function_tests.test_resource_utilization_object import TestResourceUtilizationObject
from tests.common_tests.function_tests.test_threshold_selection import TestThresholdSelection
from tests.common_tests.test_doc_examples import TestCommonDocsExamples
from tests.common_tests.test_tp_model import TargetPlatformModelingTest, OpsetTest, QCOptionsTest, FusingTest, \
    TPModelInputOutputTests

found_tf = importlib.util.find_spec("tensorflow") is not None
if found_tf:
    import tensorflow as tf
    # MCT doesn't support TensorFlow version 2.16 or higher
    if version.parse(tf.__version__) >= version.parse("2.16"):
        found_tf = False
found_pytorch = importlib.util.find_spec("torch") is not None and importlib.util.find_spec(
    "torchvision") is not None

if found_tf:
    from tests.keras_tests.xquant_tests.test_xquant_end2end import BaseTestEnd2EndKerasXQuant
    from tests.keras_tests.function_tests.test_activation_quantization_functions import TestActivationQuantizationFunctions as TestActivationQuantizationFunctionsKeras
    from tests.keras_tests.function_tests.test_custom_layer import TestCustomLayer
    from tests.keras_tests.function_tests.test_hessian_info_calculator import TestHessianInfoCalculatorWeights, \
        TestHessianInfoCalculatorActivation
    from tests.keras_tests.function_tests.test_hessian_service import TestHessianService
    from tests.keras_tests.feature_networks_tests.test_features_runner import FeatureNetworkTest
    from tests.keras_tests.function_tests.test_quantization_configurations import TestQuantizationConfigurations
    from tests.keras_tests.non_parallel_tests.test_tensorboard_writer import TestFileLogger
    from tests.keras_tests.function_tests.test_lut_quanitzer_params import TestLUTQuantizerParams
    from tests.keras_tests.function_tests.test_lut_activation_quanitzer_params import TestLUTActivationsQuantizerParams
    from tests.keras_tests.function_tests.test_lut_activation_quanitzer_fake_quant import TestLUTQuantizerFakeQuant
    from tests.keras_tests.non_parallel_tests.test_lp_search_bitwidth import TestLpSearchBitwidth, \
        TestSearchBitwidthConfiguration
    from tests.keras_tests.function_tests.test_bn_info_collection import TestBNInfoCollection
    from tests.keras_tests.graph_tests.test_graph_reading import TestGraphReading
    from tests.keras_tests.layer_tests.test_layers_runner import LayerTest as TFLayerTest
    from tests.keras_tests.function_tests.test_symmetric_threshold_selection_weights import \
        TestSymmetricThresholdSelectionWeights
    from tests.keras_tests.function_tests.test_uniform_quantize_tensor import TestUniformQuantizeTensor
    from tests.keras_tests.function_tests.test_uniform_range_selection_weights import TestUniformRangeSelectionWeights
    from tests.keras_tests.non_parallel_tests.test_keras_tp_model import TestKerasTPModel
    from tests.keras_tests.function_tests.test_sensitivity_metric_interest_points import \
        TestSensitivityMetricInterestPoints
    from tests.keras_tests.function_tests.test_weights_activation_split_substitution import TestWeightsActivationSplit
    from tests.keras_tests.function_tests.test_activation_weights_composition_substitution import \
        TestActivationWeightsComposition
    from tests.keras_tests.function_tests.test_graph_max_cut import TestGraphMaxCut
    from tests.keras_tests.function_tests.test_sensitivity_eval_non_suppoerted_output import \
        TestSensitivityEvalWithNonSupportedOutputNodes
    from tests.keras_tests.function_tests.test_set_layer_to_bitwidth import TestKerasSetLayerToBitwidth
    from tests.keras_tests.function_tests.test_export_keras_fully_quantized_model import TestKerasFakeQuantExporter
    from tests.keras_tests.function_tests.test_resource_utilization_data import TestResourceUtilizationData
    from tests.keras_tests.exporter_tests.test_runner import ExporterTestsRunner
    from tests.keras_tests.function_tests.test_get_gptq_config import TestGetGPTQConfig
    from tests.keras_tests.function_tests.test_gptq_loss_functions import TestGPTQLossFunctions
    from tests.keras_tests.trainable_infrastructure_tests.test_keras_trainable_infra_runner import \
        KerasTrainableInfrastructureTestRunner
    from tests.keras_tests.function_tests.test_gptq_soft_quantizer import TestGPTQSoftQuantizer as keras_gptq_soft_quantizer_test
    from tests.keras_tests.function_tests.test_activation_quantization_holder_gptq import TestGPTQModelBuilderWithActivationHolder
    from tests.keras_tests.data_generation_tests.test_keras_data_generation_runner import KerasDataGenerationTestRunner
    from tests.keras_tests.pruning_tests.test_memory_calculator import TestParameterCounter
    from tests.keras_tests.pruning_tests.test_pretrained_models import PruningPretrainedModelsTest
    from tests.keras_tests.pruning_tests.feature_networks.test_pruning_feature_networks import PruningFeatureNetworksTest
    from tests.keras_tests.function_tests.test_hmse_error_method import TestParamSelectionWithHMSE
    from tests.keras_tests.data_generation_tests.test_scheduler_step import TestReduceLROnPlateau
    from tests.keras_tests.function_tests.test_node_quantization_configurations import TestNodeQuantizationConfigurations
    from tests.keras_tests.function_tests.test_quant_config_filtering import TestKerasQuantConfigFiltering

if found_pytorch:
    from tests.pytorch_tests.xquant_tests.test_xquant_end2end import BaseTestEnd2EndPytorchXQuant
    from tests.pytorch_tests.function_tests.test_activation_quantization_functions import TestActivationQuantizationFunctions as TestActivationQuantizationFunctionsPytorch
    from tests.pytorch_tests.function_tests.test_torch_utils import TestTorchUtils
    from tests.pytorch_tests.function_tests.test_device_manager import TestDeviceManager
    from tests.pytorch_tests.layer_tests.test_layers_runner import LayerTest as TorchLayerTest
    from tests.pytorch_tests.model_tests.test_feature_models_runner import FeatureModelsTestRunner
    # from tests.pytorch_tests.model_tests.test_models_runner import ModelTest
    from tests.pytorch_tests.function_tests.test_function_runner import FunctionTestRunner
    from tests.pytorch_tests.function_tests.test_pytorch_tp_model import TestPytorchTPModel
    from tests.pytorch_tests.trainable_infrastructure_tests.test_pytorch_trainable_infra_runner import PytorchTrainableInfrastructureTestRunner
    from tests.pytorch_tests.function_tests.test_gptq_soft_quantizer import TestGPTQSoftQuantizer as pytorch_gptq_soft_quantier_test
    from tests.pytorch_tests.function_tests.test_activation_quantization_holder_gptq import \
        TestGPTQModelBuilderWithActivationHolder as TestGPTQModelBuilderWithActivationHolderPytorch
    from tests.pytorch_tests.exporter_tests.test_runner import PytorchExporterTestsRunner
    from tests.pytorch_tests.data_generation_tests.test_pytorch_data_generation_runner import PytorchDataGenerationTestRunner
    from tests.pytorch_tests.graph_tests.test_fx_errors import TestGraphReading
    from tests.pytorch_tests.pruning_tests.feature_networks.test_pruning_feature_networks import PruningFeatureNetworksTest
    from tests.pytorch_tests.exporter_tests.test_exporting_qat_models import TestExportingQATModelTorchscript
    from tests.pytorch_tests.function_tests.test_quant_config_filtering import TestTorchQuantConfigFiltering

if __name__ == '__main__':
    # -----------------  Load all the test cases
    suiteList = []
    suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestHistogramCollector))
    suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestCollectorsManipulations))
    suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestThresholdSelection))
    suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TPModelInputOutputTests))
    suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TargetPlatformModelingTest))
    suiteList.append(unittest.TestLoader().loadTestsFromTestCase(OpsetTest))
    suiteList.append(unittest.TestLoader().loadTestsFromTestCase(QCOptionsTest))
    suiteList.append(unittest.TestLoader().loadTestsFromTestCase(FusingTest))
    suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestCommonDocsExamples))
    suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestResourceUtilizationObject))
    suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestEdgeMatcher))
    suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestLogger))
    suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestImmutableClass))

    # Add TF tests only if tensorflow is installed
    if found_tf:
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(BaseTestEnd2EndKerasXQuant))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestActivationQuantizationFunctionsKeras))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestReduceLROnPlateau))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestCustomLayer))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestParameterCounter))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(PruningPretrainedModelsTest))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(PruningFeatureNetworksTest))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestHessianInfoCalculatorWeights))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestHessianInfoCalculatorActivation))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestHessianService))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestGPTQModelBuilderWithActivationHolder))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(ExporterTestsRunner))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestSensitivityMetricInterestPoints))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestQuantizationConfigurations))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestNodeQuantizationConfigurations))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(FeatureNetworkTest))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestLpSearchBitwidth))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestSearchBitwidthConfiguration))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestBNInfoCollection))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestLUTQuantizerParams))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestLUTActivationsQuantizerParams))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestLUTQuantizerFakeQuant))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestGraphReading))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestSymmetricThresholdSelectionWeights))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestUniformQuantizeTensor))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestUniformRangeSelectionWeights))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestKerasTPModel))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestWeightsActivationSplit))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestActivationWeightsComposition))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestGraphMaxCut))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestKerasSetLayerToBitwidth))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestSensitivityEvalWithNonSupportedOutputNodes))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestKerasFakeQuantExporter))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestResourceUtilizationData))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestFileLogger))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestGetGPTQConfig))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestGPTQLossFunctions))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(KerasTrainableInfrastructureTestRunner))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(keras_gptq_soft_quantizer_test))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TFLayerTest))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(KerasDataGenerationTestRunner))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestParamSelectionWithHMSE))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestKerasQuantConfigFiltering))

    if found_pytorch:
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(BaseTestEnd2EndPytorchXQuant))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestActivationQuantizationFunctionsPytorch))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestTorchUtils))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestDeviceManager))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestGPTQModelBuilderWithActivationHolderPytorch))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TorchLayerTest))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(FeatureModelsTestRunner))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(FunctionTestRunner))
        # Exporter test of pytorch must have ONNX installed
        # suiteList.append(unittest.TestLoader().loadTestsFromName('test_mobilenet_v2', ModelTest))
        # suiteList.append(unittest.TestLoader().loadTestsFromName('test_mobilenet_v3', ModelTest))
        # suiteList.append(unittest.TestLoader().loadTestsFromName('test_efficientnet_b0', ModelTest))
        # suiteList.append(unittest.TestLoader().loadTestsFromName('test_resnet18', ModelTest))
        # suiteList.append(unittest.TestLoader().loadTestsFromName('test_shufflenet_v2_x1_0', ModelTest))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestPytorchTPModel))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(pytorch_gptq_soft_quantier_test))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(PytorchTrainableInfrastructureTestRunner))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(PytorchExporterTestsRunner))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(PytorchDataGenerationTestRunner))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestGraphReading))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(PruningFeatureNetworksTest))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestExportingQATModelTorchscript))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestTorchQuantConfigFiltering))

    # ----------------   Join them together and run them
    comboSuite = unittest.TestSuite(suiteList)
    unittest.TextTestRunner(verbosity=0).run(comboSuite)

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

#  ----------------  Individual test suites
from tests.common_tests.function_tests.test_histogram_collector import TestHistogramCollector
from tests.common_tests.function_tests.test_collectors_manipulation import TestCollectorsManipulations
from tests.common_tests.function_tests.test_threshold_selection import TestThresholdSelection
from tests.common_tests.function_tests.test_folder_image_loader import TestFolderLoader
from tests.common_tests.test_tp_model import TargetPlatformModelingTest, OpsetTest, QCOptionsTest, FusingTest

found_tf = importlib.util.find_spec("tensorflow") is not None and importlib.util.find_spec(
    "tensorflow_model_optimization") is not None
found_pytorch = importlib.util.find_spec("torch") is not None and importlib.util.find_spec(
    "torchvision") is not None

if found_tf:
    import tensorflow as tf
    from tests.keras_tests.feature_networks_tests.test_features_runner import FeatureNetworkTest
    from tests.keras_tests.function_tests.test_quantization_configurations import TestQuantizationConfigurations
    from tests.keras_tests.function_tests.test_tensorboard_writer import TestLogger
    from tests.keras_tests.function_tests.test_lut_quanitzer_params import TestLUTQuantizerParams
    from tests.keras_tests.function_tests.test_lp_search_bitwidth import TestLpSearchBitwidth, \
        TestSearchBitwidthConfiguration
    from tests.keras_tests.function_tests.test_bn_info_collection import TestBNInfoCollection
    from tests.keras_tests.graph_tests.test_graph_reading import TestGraphReading
    from tests.keras_tests.graph_tests.test_graph_quantization_and_export import TestTFLiteExport
    from tests.keras_tests.layer_tests.test_layers_runner import LayerTest as TFLayerTest
    from tests.keras_tests.function_tests.test_symmetric_threshold_selection_weights import \
        TestSymmetricThresholdSelectionWeights
    from tests.keras_tests.function_tests.test_uniform_quantize_tensor import TestUniformQuantizeTensor
    from tests.keras_tests.function_tests.test_uniform_range_selection_weights import TestUniformRangeSelectionWeights
    from tests.keras_tests.test_keras_tp_model import TestKerasTPModel
    from tests.keras_tests.function_tests.test_sensitivity_metric_interest_points import \
        TestSensitivityMetricInterestPoints


if found_pytorch:
    from tests.pytorch_tests.layer_tests.test_layers_runner import LayerTest as TorchLayerTest
    from tests.pytorch_tests.model_tests.test_feature_models_runner import FeatureModelsTestRunner
    from tests.pytorch_tests.model_tests.test_models_runner import ModelTest
    from tests.pytorch_tests.function_tests.test_function_runner import FunctionTestRunner
    from tests.pytorch_tests.test_pytorch_tp_model import TestPytorchTPModel


if __name__ == '__main__':
    # -----------------  Load all the test cases
    suiteList = []
    suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestHistogramCollector))
    suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestCollectorsManipulations))
    suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestFolderLoader))
    suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestThresholdSelection))
    suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TargetPlatformModelingTest))
    suiteList.append(unittest.TestLoader().loadTestsFromTestCase(OpsetTest))
    suiteList.append(unittest.TestLoader().loadTestsFromTestCase(QCOptionsTest))
    suiteList.append(unittest.TestLoader().loadTestsFromTestCase(FusingTest))

    # Add TF tests only if tensorflow is installed
    if found_tf:
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestSensitivityMetricInterestPoints))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestQuantizationConfigurations))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(FeatureNetworkTest))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestLogger))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestLpSearchBitwidth))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestSearchBitwidthConfiguration))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestBNInfoCollection))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestLUTQuantizerParams))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestGraphReading))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestTFLiteExport))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestSymmetricThresholdSelectionWeights))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestUniformQuantizeTensor))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestUniformRangeSelectionWeights))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestKerasTPModel))

        # Keras test layers are supported in TF2.6 or higher versions
        if tf.__version__ >= "2.6":
            suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TFLayerTest))

    if found_pytorch:
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TorchLayerTest))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(FeatureModelsTestRunner))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(FunctionTestRunner))
        suiteList.append(unittest.TestLoader().loadTestsFromName('test_mobilenet_v2', ModelTest))
        suiteList.append(unittest.TestLoader().loadTestsFromName('test_mobilenet_v3', ModelTest))
        suiteList.append(unittest.TestLoader().loadTestsFromName('test_efficientnet_b0', ModelTest))
        suiteList.append(unittest.TestLoader().loadTestsFromName('test_resnet18', ModelTest))
        suiteList.append(unittest.TestLoader().loadTestsFromName('test_shufflenet_v2_x1_0', ModelTest))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestPytorchTPModel))

    # ----------------   Join them together and run them
    comboSuite = unittest.TestSuite(suiteList)
    unittest.TextTestRunner(verbosity=0).run(comboSuite)

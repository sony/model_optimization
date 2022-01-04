# Copyright 2021 Sony Semiconductors Israel, Inc. All rights reserved.
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

found_tf = importlib.util.find_spec("tensorflow") is not None and importlib.util.find_spec(
    "tensorflow_model_optimization") is not None
if found_tf:
    import tensorflow as tf
    from tests.keras_tests.feature_networks_tests.test_features_runner import FeatureNetworkTest
    from tests.keras_tests.function_tests.test_quantization_configurations import TestQuantizationConfigurations
    from tests.keras_tests.function_tests.test_tensorboard_writer import TestLogger
    from tests.keras_tests.function_tests.test_lut_quanitzer_params import TestLUTQuantizerParams
    from tests.keras_tests.function_tests.test_lp_search_bitwidth import TestLpSearchBitwidth, \
        TestSearchBitwidthConfiguration
    from tests.keras_tests.graph_tests.test_graph_reading import TestGraphReading
    from tests.keras_tests.graph_tests.test_graph_quantization_and_export import TestTFLiteExport
    from tests.keras_tests.layer_tests.test_layers_runner import LayerTest

if __name__ == '__main__':
    # -----------------  Load all the test cases
    suiteList = []
    suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestHistogramCollector))
    suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestCollectorsManipulations))
    suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestFolderLoader))
    suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestThresholdSelection))

    # Add TF tests only if tensorflow is installed
    if found_tf:
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestQuantizationConfigurations))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(FeatureNetworkTest))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestLogger))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestLpSearchBitwidth))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestSearchBitwidthConfiguration))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestLUTQuantizerParams))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestGraphReading))
        suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestTFLiteExport))
        # Keras test layers are supported in TF2.6 or higher versions
        if tf.__version__ >= "2.6":
            suiteList.append(unittest.TestLoader().loadTestsFromTestCase(LayerTest))

    # ----------------   Join them together ane run them
    comboSuite = unittest.TestSuite(suiteList)
    unittest.TextTestRunner(verbosity=0).run(comboSuite)

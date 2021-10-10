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
import unittest

#  ----------------  Individual test suites
from tests.feature_networks_tests.test_features_runner import FeatureNetworkTest
from tests.function_tests.test_histogram_collector import TestHistogramCollector
from tests.function_tests.test_quantization_configurations import TestQuantizationConfigurations
from tests.function_tests.test_logger import TestLogger

if __name__ == '__main__':
    # -----------------  Load all the test cases
    suiteList = []
    suiteList.append(unittest.TestLoader().loadTestsFromTestCase(FeatureNetworkTest))
    suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestHistogramCollector))
    suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestQuantizationConfigurations))
    suiteList.append(unittest.TestLoader().loadTestsFromTestCase(TestLogger))

    # ----------------   Join them together ane run them
    comboSuite = unittest.TestSuite(suiteList)
    unittest.TextTestRunner(verbosity=0).run(comboSuite)

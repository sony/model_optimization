# Copyright 2022 Sony Semiconductors Israel, Inc. All rights reserved.
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

from tests.pytorch_tests.function_tests.bn_info_collection_test import BNInfoCollectionTest, \
    Conv2D2BNInfoCollectionTest, Conv2DBNChainInfoCollectionTest, BNChainInfoCollectionTest, \
    BNLayerInfoCollectionTest, INP2BNInfoCollectionTest
from tests.pytorch_tests.function_tests.kpi_data_test import TestKPIDataBasicAllBitwidth, \
    TestKPIDataBasicPartialBitwidth, TestKPIDataComplexPartialBitwidth, TestKPIDataComplesAllBitwidth
from tests.pytorch_tests.function_tests.model_gradients_test import ModelGradientsBasicModelTest, \
    ModelGradientsCalculationTest, ModelGradientsAdvancedModelTest, ModelGradientsOutputReplacementTest


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

    def test_kpi_data_basic_all(self):
        """
        This test checks the KPI data Pytorch API.
        """
        TestKPIDataBasicAllBitwidth(self).run_test()

    def test_kpi_data_basic_partial(self):
        """
        This test checks the KPI data Pytorch API.
        """
        TestKPIDataBasicPartialBitwidth(self).run_test()

    def test_kpi_data_complex_all(self):
        """
        This test checks the KPI data Pytorch API.
        """
        TestKPIDataComplesAllBitwidth(self).run_test()

    def test_kpi_data_complex_partial(self):
        """
        This test checks the KPI data Pytorch API.
        """
        TestKPIDataComplexPartialBitwidth(self).run_test()

    def test_model_gradients(self):
        """
        This test checks the Model Gradients Pytorch computation.
        """
        ModelGradientsBasicModelTest(self).run_test()
        ModelGradientsCalculationTest(self).run_test()
        ModelGradientsAdvancedModelTest(self).run_test()
        ModelGradientsOutputReplacementTest(self).run_test()


if __name__ == '__main__':
    unittest.main()

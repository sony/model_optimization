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

from tests.pytorch_tests.model_tests.feature_models.add_net_test import AddNetTest
from tests.pytorch_tests.model_tests.feature_models.add_same_test import AddSameNetTest
from tests.pytorch_tests.model_tests.feature_models.bn_folding_test import BNFoldingNetTest
from tests.pytorch_tests.model_tests.feature_models.layer_name_test import ReuseNameNetTest
from tests.pytorch_tests.model_tests.feature_models.lut_quantizer_test import LUTQuantizerTest
from tests.pytorch_tests.model_tests.feature_models.mixed_precision_test import MixedPercisionSearch8Bit, \
    MixedPercisionSearch2Bit
from tests.pytorch_tests.model_tests.feature_models.multiple_output_nodes_multiple_tensors_test import \
    MultipleOutputsMultipleTensorsNetTest
from tests.pytorch_tests.model_tests.feature_models.multiple_outputs_node_test import MultipleOutputsNetTest
from tests.pytorch_tests.model_tests.feature_models.output_in_the_middle_test import OutputInTheMiddleNetTest
from tests.pytorch_tests.model_tests.feature_models.reuse_layer_net_test import ReuseLayerNetTest
from tests.pytorch_tests.model_tests.feature_models.shift_negative_activation_test import ShiftNegaviteActivationNetTest
from tests.pytorch_tests.model_tests.feature_models.split_concat_net_test import SplitConcatNetTest


class FeatureModelsTestRunner(unittest.TestCase):

    def test_add_net(self):
        """
        This tests check the addition and subtraction operations.
        Both with different layers and with constants.
        """
        AddNetTest(self).run_test()

    def test_add_same(self):
        """
        This test checks the special case of addition operation with the same input.
        """
        AddSameNetTest(self, float_reconstruction_error=1e-6).run_test()

    def test_bn_folding(self):
        """
        This test checks the BatchNorm folding feature, plus adding a residual connection.
        """
        BNFoldingNetTest(self).run_test()

    def test_layer_name(self):
        """
        This test checks that we build a correct graph and correctly reconstruct the model
        given the fact that we reuse nodes and abuse the naming convention of fx (if we resuse
        "conv1" it will be displayed as "conv1_1". So we intentionally add a node named "conv1_1").
        """
        ReuseNameNetTest(self).run_test()

    def test_lut_quantizer(self):
        """
        This test checks multiple features:
        1. That the LUT quantizer quantizes the weights differently than than the Power-of-two quantizer
        2. That the Network Editor works

        In this test we set the weights of 3 conv2d operator to be the same. We set the quantization method
        to "Power-of-two". With the Network Editor we change the quantization method of "conv1" to "LUT quantizer".
        We check that the weights have different values for conv1 and conv2, and that conv2 and conv3 have the same
        values.
        """
        LUTQuantizerTest(self).run_test()

    def test_multiple_output_nodes_multiple_tensors(self):
        """
        This test checks that we support the connecting the input tensor to several layers
        and taking them as outputs
        """
        MultipleOutputsMultipleTensorsNetTest(self).run_test()

    def test_multiple_outputs_node(self):
        """
        This test checks that we support multiple outputs in the network and multiple
        outputs from a node
        """
        MultipleOutputsNetTest(self).run_test()

    def test_output_in_the_middle(self):
        """
        This test checks:
        That we support taking outputs from the middle of the model.
        """
        OutputInTheMiddleNetTest(self).run_test()

    def test_reuse_layer_net(self):
        """
        This test checks:
        The reuse of a layer in a model.
        """
        ReuseLayerNetTest(self).run_test()

    def test_shift_negative_activation_net(self):
        """
        This test checks the shift negative activation feature.
        """
        ShiftNegaviteActivationNetTest(self).run_test(seed=3)

    def test_split_concat_net(self):
        """
        This test checks:
        1. The "split" and "concat" operations.
        2. Nodes with multiple outputs and multiple inputs
        """
        SplitConcatNetTest(self).run_test()

    def test_mixed_precision_8bit(self):
        """
        This test checks the Mixed Precision search.
        """
        MixedPercisionSearch8Bit(self).run_test()

    def test_mixed_precision_2bit(self):
        """
        This test checks the Mixed Precision search.
        """
        MixedPercisionSearch2Bit(self).run_test()


if __name__ == '__main__':
    unittest.main()
# Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
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

import torch.nn
from tests.pytorch_tests.pruning_tests.feature_networks.network_tests.conv2d_conv2dtranspose_pruning_test import Conv2dConv2dTransposePruningTest
from tests.pytorch_tests.pruning_tests.feature_networks.network_tests.conv2d_pruning_test import Conv2DPruningTest
from tests.pytorch_tests.pruning_tests.feature_networks.network_tests.conv2dtranspose_conv2d_pruning_test import \
    Conv2dTransposeConv2dPruningTest
from tests.pytorch_tests.pruning_tests.feature_networks.network_tests.conv2dtranspose_pruning_test import \
    Conv2dTransposePruningTest
from tests.pytorch_tests.pruning_tests.feature_networks.network_tests.linear_pruning_test import LinearPruningTest

class PruningFeatureNetworksTest(unittest.TestCase):

    def test_conv2d_pruning(self):
        Conv2DPruningTest(self, use_bn=False).run_test()
        Conv2DPruningTest(self, use_bn=True).run_test()
        Conv2DPruningTest(self, use_bn=True, activation_layer=torch.nn.ReLU()).run_test()
        Conv2DPruningTest(self, use_bn=True, activation_layer=torch.nn.Softmax()).run_test()
        Conv2DPruningTest(self, use_bn=True, activation_layer=torch.nn.PReLU()).run_test()
        Conv2DPruningTest(self, simd=2).run_test()
        Conv2DPruningTest(self, use_bn=True, simd=2).run_test()
        Conv2DPruningTest(self, use_bn=True, activation_layer=torch.nn.ReLU(), simd=2).run_test()
        Conv2DPruningTest(self, use_bn=True, activation_layer=torch.nn.Softmax(), simd=2).run_test()
        Conv2DPruningTest(self, use_bn=True, activation_layer=torch.nn.PReLU(), simd=2).run_test()

        # Use dummy LFH
        Conv2DPruningTest(self, use_constant_importance_metric=False).run_test()
        Conv2DPruningTest(self, simd=2, use_constant_importance_metric=False).run_test()

    def test_linear_pruning(self):
        LinearPruningTest(self).run_test()
        # LinearPruningTest(self, use_bn=True).run_test()
        LinearPruningTest(self, activation_layer=torch.nn.ReLU()).run_test()
        LinearPruningTest(self, activation_layer=torch.nn.Softmax()).run_test()
        LinearPruningTest(self, activation_layer=torch.nn.PReLU()).run_test()
        LinearPruningTest(self, simd=2).run_test()
        # LinearPruningTest(self, use_bn=True, simd=2).run_test()
        LinearPruningTest(self, activation_layer=torch.nn.ReLU(), simd=2).run_test()
        LinearPruningTest(self, activation_layer=torch.nn.Softmax(), simd=2).run_test()
        LinearPruningTest(self, activation_layer=torch.nn.PReLU(), simd=2).run_test()

        # Use dummy LFH
        LinearPruningTest(self, use_constant_importance_metric=False).run_test()
        LinearPruningTest(self, simd=2, use_constant_importance_metric=False).run_test()

    def test_conv2dtranspose_pruning(self):
        Conv2dTransposePruningTest(self).run_test()
        Conv2dTransposePruningTest(self, use_bn=True).run_test()
        Conv2dTransposePruningTest(self, use_bn=True, activation_layer=torch.nn.ReLU()).run_test()
        Conv2dTransposePruningTest(self, use_bn=True, activation_layer=torch.nn.Softmax()).run_test()
        Conv2dTransposePruningTest(self, use_bn=True, activation_layer=torch.nn.PReLU()).run_test()
        Conv2dTransposePruningTest(self, simd=2).run_test()
        Conv2dTransposePruningTest(self, use_bn=True, simd=2).run_test()
        Conv2dTransposePruningTest(self, use_bn=True, activation_layer=torch.nn.ReLU(), simd=2).run_test()
        Conv2dTransposePruningTest(self, use_bn=True, activation_layer=torch.nn.Softmax(), simd=2).run_test()
        Conv2dTransposePruningTest(self, use_bn=True, activation_layer=torch.nn.PReLU(), simd=2).run_test()
        # Use dummy LFH
        Conv2dTransposePruningTest(self, use_constant_importance_metric=False).run_test()
        Conv2dTransposePruningTest(self, simd=2, use_constant_importance_metric=False).run_test()


    def test_conv2d_conv2dtranspose_pruning(self):
        Conv2dTransposeConv2dPruningTest(self).run_test()
        Conv2dTransposeConv2dPruningTest(self, use_bn=True).run_test()
        Conv2dTransposeConv2dPruningTest(self, use_bn=True, activation_layer=torch.nn.ReLU()).run_test()
        Conv2dTransposeConv2dPruningTest(self, use_bn=True, activation_layer=torch.nn.Softmax()).run_test()
        Conv2dTransposeConv2dPruningTest(self, use_bn=True, activation_layer=torch.nn.PReLU()).run_test()
        Conv2dTransposeConv2dPruningTest(self, simd=2).run_test()
        Conv2dTransposeConv2dPruningTest(self, use_bn=True, simd=2).run_test()
        Conv2dTransposeConv2dPruningTest(self, use_bn=True, activation_layer=torch.nn.ReLU(), simd=2).run_test()
        Conv2dTransposeConv2dPruningTest(self, use_bn=True, activation_layer=torch.nn.Softmax(), simd=2).run_test()
        Conv2dTransposeConv2dPruningTest(self, use_bn=True, activation_layer=torch.nn.PReLU(), simd=2).run_test()
        # Use dummy LFH
        Conv2dTransposeConv2dPruningTest(self, use_constant_importance_metric=False).run_test()
        Conv2dTransposeConv2dPruningTest(self, simd=2, use_constant_importance_metric=False).run_test()

    def test_conv2dtranspose_conv2d_pruning(self):
        Conv2dConv2dTransposePruningTest(self).run_test()
        Conv2dConv2dTransposePruningTest(self, use_bn=True).run_test()
        Conv2dConv2dTransposePruningTest(self, use_bn=True, activation_layer=torch.nn.ReLU()).run_test()
        Conv2dConv2dTransposePruningTest(self, use_bn=True, activation_layer=torch.nn.Softmax()).run_test()
        Conv2dConv2dTransposePruningTest(self, use_bn=True, activation_layer=torch.nn.PReLU()).run_test()
        Conv2dConv2dTransposePruningTest(self, simd=2).run_test()
        Conv2dConv2dTransposePruningTest(self, use_bn=True, simd=2).run_test()
        Conv2dConv2dTransposePruningTest(self, use_bn=True, activation_layer=torch.nn.ReLU(), simd=2).run_test()
        Conv2dConv2dTransposePruningTest(self, use_bn=True, activation_layer=torch.nn.Softmax(), simd=2).run_test()
        Conv2dConv2dTransposePruningTest(self, use_bn=True, activation_layer=torch.nn.PReLU(), simd=2).run_test()
        # Use dummy LFH
        Conv2dConv2dTransposePruningTest(self, use_constant_importance_metric=False).run_test()
        Conv2dConv2dTransposePruningTest(self, simd=2, use_constant_importance_metric=False).run_test()


if __name__ == '__main__':
    unittest.main()
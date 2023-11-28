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

from tests.keras_tests.pruning_tests.networks_tests.conv2d_pruning_test import Conv2DPruningTest
import numpy as np

from tests.keras_tests.pruning_tests.networks_tests.conv2dtranspose_pruning_test import Conv2DTransposePruningTest
from tests.keras_tests.pruning_tests.networks_tests.dense_pruning_test import DensePruningTest
import keras

layers = keras.layers

class PruningNetworksTest(unittest.TestCase):

    def test_conv2d_pruning(self):
        target_crs = np.linspace(0.5, 1, 5)
        for cr in target_crs:
            Conv2DPruningTest(self, target_cr=cr).run_test()
            Conv2DPruningTest(self, target_cr=cr, use_bn=True).run_test()
            Conv2DPruningTest(self, target_cr=cr, use_bn=True, activation_layer=layers.ReLU()).run_test()
            Conv2DPruningTest(self, target_cr=cr, use_bn=True, activation_layer=layers.Softmax()).run_test()
            Conv2DPruningTest(self, target_cr=cr, use_bn=True, activation_layer=layers.PReLU()).run_test()


    def test_dense_pruning(self):
        target_crs = np.linspace(0.5, 1, 5)
        for cr in target_crs:
            DensePruningTest(self, target_cr=cr).run_test()
            DensePruningTest(self, target_cr=cr, use_bn=True).run_test()
            DensePruningTest(self, target_cr=cr, use_bn=True, activation_layer=layers.ReLU()).run_test()
            DensePruningTest(self, target_cr=cr, use_bn=True, activation_layer=layers.Softmax()).run_test()
            DensePruningTest(self, target_cr=cr, use_bn=True, activation_layer=layers.PReLU()).run_test()


    def test_conv2dtranspose_pruning(self):
        target_crs = np.linspace(0.5, 1, 5)
        for cr in target_crs:
            Conv2DTransposePruningTest(self, target_cr=cr).run_test()
            Conv2DTransposePruningTest(self, target_cr=cr, use_bn=True).run_test()
            Conv2DTransposePruningTest(self, target_cr=cr, use_bn=True, activation_layer=layers.ReLU()).run_test()
            Conv2DTransposePruningTest(self, target_cr=cr, use_bn=True, activation_layer=layers.Softmax()).run_test()
            Conv2DTransposePruningTest(self, target_cr=cr, use_bn=True, activation_layer=layers.PReLU()).run_test()


if __name__ == '__main__':
    unittest.main()

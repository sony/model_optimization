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
import unittest
from tests.keras_tests.layer_tests.layers.activation_test import ActivationTest
from tests.keras_tests.layer_tests.layers.concat_test import ConcatTest
from tests.keras_tests.layer_tests.layers.conv2d_test import Conv2DTest
from tests.keras_tests.layer_tests.layers.crop_and_resize_test import CropAndResizeTest
from tests.keras_tests.layer_tests.layers.dense_test import DenseTest
from tests.keras_tests.layer_tests.layers.math_add_test import MathAddTest
from tests.keras_tests.layer_tests.layers.math_mul_test import MathMulTest
from tests.keras_tests.layer_tests.layers.reduce_mean_test import ReduceMeanTest
from tests.keras_tests.layer_tests.layers.reduce_max_test import ReduceMaxTest
from tests.keras_tests.layer_tests.layers.reduce_min_test import ReduceMinTest
from tests.keras_tests.layer_tests.layers.reduce_sum_test import ReduceSumTest
from tests.keras_tests.layer_tests.layers.relu_test import ReLUTest
from tests.keras_tests.layer_tests.layers.reshape_test import ReshapeTest
from tests.keras_tests.layer_tests.layers.resize_test import ResizeTest
from tests.keras_tests.layer_tests.layers.split_test import SplitTest



class LayerTest(unittest.TestCase):
    LAYERS2RUN = [ActivationTest, ConcatTest, Conv2DTest, CropAndResizeTest, DenseTest, MathAddTest, MathMulTest,
                  ReLUTest,
                  ReduceMeanTest,
                  ReduceMaxTest,
                  ReduceMinTest,
                  ResizeTest,
                  ReshapeTest,
                  ReduceSumTest,
                  SplitTest]

    def test_keras_layers(self):
        keras_layer_tests = {cls.__name__: cls for cls in self.LAYERS2RUN}
        for k, v in keras_layer_tests.items():
            # Test each one of them independently:
            with self.subTest(msg=k):
                v(unittest.TestCase()).run_test()


if __name__ == '__main__':
    unittest.main()

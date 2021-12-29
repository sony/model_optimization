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
import importlib
import os
import pkgutil
import unittest
from tests.keras_tests.layer_tests.base_keras_layer_test import BaseKerasLayerTest

# If empty -> Run all layer tests
# If not -> Run just these tests (e.g. layers=[DenseTest, ConcatTest])

layers = []

layers_pkg_rel_path = 'keras_tests/layer_tests/layers/'

class LayerTest(unittest.TestCase):

    def test_keras_layers(self):
        # Import all test cases from package 'layers':
        for (module_loader, name, ispkg) in pkgutil.iter_modules([layers_pkg_rel_path]):
            importlib.import_module(layers_pkg_rel_path.replace('/', '.') + name, __package__)
        # Get all layer tests as they subclasses of BaseKerasLayerTest
        keras_layer_tests = {cls.__name__: cls for cls in BaseKerasLayerTest.__subclasses__()}
        for k, v in keras_layer_tests.items():
            if len(layers) == 0 or k in [l.__name__ for l in layers]:
                # Test each one of them independently:
                with self.subTest(msg=k):
                    v(unittest.TestCase()).run_test()


if __name__ == '__main__':
    unittest.main()

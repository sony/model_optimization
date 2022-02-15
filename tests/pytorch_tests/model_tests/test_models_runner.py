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
from torchvision.models import mobilenet_v2, mobilenet_v3_large, efficientnet_b0, resnet18
from tests.pytorch_tests.model_tests.model_zoo_tests.base_model_test import BaseModelTest


class ModelTest(unittest.TestCase):

    def test_mobilenet_v2(self):
        BaseModelTest(self, mobilenet_v2, float_reconstruction_error=1e-4).run_test()

    def test_mobilenet_v3(self):
        BaseModelTest(self, mobilenet_v3_large, float_reconstruction_error=1e-4).run_test()

    def test_efficientnet_b0(self):
        BaseModelTest(self, efficientnet_b0, float_reconstruction_error=1e-4).run_test()

    def test_resnet18(self):
        BaseModelTest(self, resnet18, float_reconstruction_error=1e-4).run_test()


if __name__ == '__main__':
    unittest.main()

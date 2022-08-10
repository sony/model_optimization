# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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
from torchvision.models import mobilenet_v2, mobilenet_v3_large, efficientnet_b0, resnet18, shufflenet_v2_x1_0, \
    mnasnet1_0, alexnet, densenet121, googlenet, inception_v3, regnet_x_1_6gf, resnext50_32x4d, squeezenet1_0, vgg16, \
    wide_resnet50_2
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

    def test_shufflenet_v2_x1_0(self):
        BaseModelTest(self, shufflenet_v2_x1_0, float_reconstruction_error=1e-4, convert_to_fx=False).run_test()

    def test_mnasnet1_0(self):
        BaseModelTest(self, mnasnet1_0, float_reconstruction_error=1e-4).run_test()

    def test_alexnet(self):
        BaseModelTest(self, alexnet, float_reconstruction_error=1e-4).run_test()

    def test_densenet121(self):
        BaseModelTest(self, densenet121, float_reconstruction_error=1e-4).run_test()

    def test_googlenet(self):
        BaseModelTest(self, googlenet, float_reconstruction_error=1e-4).run_test()

    def test_inception_v3(self):
        BaseModelTest(self, inception_v3, float_reconstruction_error=1e-4).run_test()

    def test_regnet_x_1_6gf(self):
        BaseModelTest(self, regnet_x_1_6gf, float_reconstruction_error=1e-4).run_test()

    def test_resnext50_32x4d(self):
        BaseModelTest(self, resnext50_32x4d, float_reconstruction_error=1e-4).run_test()

    def test_squeezenet1_0(self):
        BaseModelTest(self, squeezenet1_0, float_reconstruction_error=1e-4).run_test()

    def test_vgg16(self):
        BaseModelTest(self, vgg16, float_reconstruction_error=1e-4).run_test()

    def test_wide_resnet50_2(self):
        BaseModelTest(self, wide_resnet50_2, float_reconstruction_error=1e-4).run_test()


if __name__ == '__main__':
    unittest.main()
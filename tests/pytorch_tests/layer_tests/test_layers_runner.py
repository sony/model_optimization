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

from torch.nn import Hardswish, Hardsigmoid, ReLU, ReLU6, LeakyReLU, PReLU, SiLU, Softmax, \
    Sigmoid, Softplus, Softsign, Tanh
from torch.nn.functional import hardswish, hardsigmoid, relu, relu6, leaky_relu, prelu, silu, softmax, \
    softplus, softsign
from torch.nn import UpsamplingBilinear2d, AdaptiveAvgPool2d, AvgPool2d, MaxPool2d
from torch.nn import Conv2d, ConvTranspose2d, Linear, BatchNorm2d
from torch.nn import Dropout, Flatten
from torch import add, multiply, mul, sub, flatten, reshape, split, unsqueeze, concat, cat, mean, \
    dropout, sigmoid, tanh
import operator

from tests.pytorch_tests.layer_tests.base_pytorch_layer_test import BasePytorchLayerTest
from tests.pytorch_tests.layer_tests.single_layer_models import ReshapeModel, SplitModel, ConcatModel, CatModel, \
    DropoutModel, UnsqueezeModel, MeanModel, PReluModel


class LayerTest(unittest.TestCase):

    def test_activation(self):
        BasePytorchLayerTest(self,
                             [
                                 Hardswish(),
                                 Hardsigmoid(),
                                 ReLU(),
                                 ReLU6(),
                                 LeakyReLU(),
                                 PReLU(),
                                 SiLU(),
                                 Softmax(),
                                 Sigmoid(),
                                 Softplus(),
                                 Softsign(),
                                 Tanh(),
                                 ReLU(),
                             ]).run_test()

    def test_activation_functional(self):
        BasePytorchLayerTest(self,
                             [
                                 hardswish,
                                 hardsigmoid,
                                 relu,
                                 relu6,
                                 leaky_relu,
                                 silu,
                                 softmax,
                                 sigmoid,
                                 softplus,
                                 softsign,
                                 tanh
                             ]).run_test()

    def test_resampling_ops(self):
        BasePytorchLayerTest(self,
                             [
                                 UpsamplingBilinear2d(2),
                                 AdaptiveAvgPool2d(4),
                                 AvgPool2d(2),
                                 MaxPool2d(2)
                             ]).run_test()

    def test_nn_ops(self):
        BasePytorchLayerTest(self,
                             [
                                 Dropout(),
                                 Flatten(),
                             ]).run_test()

    def test_torch_operations(self):
        BasePytorchLayerTest(self,
                             [
                                 flatten,
                                 ReshapeModel(),
                                 SplitModel(),
                                 ConcatModel(),
                                 CatModel(),
                                 DropoutModel(),
                                 UnsqueezeModel(),
                                 MeanModel(),
                             ]).run_test()

    def test_operations(self):
        BasePytorchLayerTest(self,
                             [
                                 add,
                                 multiply,
                                 mul,
                                 sub,
                                 operator.add,
                                 operator.mul,
                                 operator.sub
                             ],
                             num_of_inputs=2).run_test()

    def test_conv2d_ops(self):
        BasePytorchLayerTest(self,
                             [
                                 Conv2d(8, 12, 1, bias=False),
                                 Conv2d(8, 8, 3, bias=True),
                                 Conv2d(8, 8, 1, bias=False, groups=8),# dw conv
                                 Conv2d(8, 8, 3, bias=True, groups=8),# dw conv
                                 Conv2d(8, 8, 1, bias=False, groups=4),# group conv
                                 Conv2d(8, 12, 3, bias=True, groups=2),# group conv
                             ], input_shape=(8, 10, 10)).run_test()

    def test_conv_transpose2d_ops(self):
        BasePytorchLayerTest(self,
                             [
                                 ConvTranspose2d(8, 8, 3, bias=True),
                                 ConvTranspose2d(8, 12, 3, bias=False),
                                 BatchNorm2d(8)
                             ], input_shape=(8, 10, 10)).run_test()

    def test_linear_ops(self):
        BasePytorchLayerTest(self,
                             [
                                 Linear(10, 12, bias=True),
                                 Linear(10, 8, bias=False),
                             ], input_shape=(10,)).run_test()

    def test_more_layers(self):
        BasePytorchLayerTest(self,
                             [
                                 ReshapeModel(),
                                 SplitModel(),
                                 ConcatModel(),
                                 CatModel(),
                                 DropoutModel(),
                                 UnsqueezeModel(),
                                 MeanModel(),
                                 # PReluModel()
                             ]).run_test()

if __name__ == '__main__':
    unittest.main()

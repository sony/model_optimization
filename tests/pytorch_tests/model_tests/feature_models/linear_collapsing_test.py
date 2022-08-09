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
from abc import ABC
import torch.nn as nn
import numpy as np
import model_compression_toolkit as mct
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor, torch_tensor_to_numpy
from tests.pytorch_tests.model_tests.base_pytorch_feature_test import BasePytorchFeatureNetworkTest
from tests.common_tests.helpers.generate_test_tp_model import generate_test_tp_model
from model_compression_toolkit.core.tpc_models.default_tpc.latest import generate_pytorch_tpc
from tests.common_tests.helpers.tensors_compare import cosine_similarity

tp = mct.target_platform


class BaseConv2DCollapsingTest(BasePytorchFeatureNetworkTest, ABC):

    def __init__(self, unit_test):
        super().__init__(unit_test=unit_test, input_shape=(16,32,32))

    def get_tpc(self):
        tp = generate_test_tp_model({'weights_n_bits': 32,
                                     'activation_n_bits': 32,
                                     'enable_weights_quantization': False,
                                     'enable_activation_quantization': False})
        return generate_pytorch_tpc(name="linear_collapsing_test", tp_model=tp)

    def get_quantization_config(self):
        return mct.QuantizationConfig(mct.QuantizationErrorMethod.NOCLIPPING, mct.QuantizationErrorMethod.NOCLIPPING,
                                      False, False, True)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        in_torch_tensor = to_torch_tensor(input_x[0])
        y = float_model(in_torch_tensor)
        y_hat = quantized_model(in_torch_tensor)
        self.unit_test.assertTrue(y.shape == y_hat.shape, msg=f'out shape is not as expected!')
        # quantized_model has additional placeholder layer, so we add -1
        self.unit_test.assertTrue(len(list(quantized_model.modules())) - 1 < len(list(float_model.modules())), msg=f'fail number of layers should decrease!')
        cs = cosine_similarity(torch_tensor_to_numpy(y), torch_tensor_to_numpy(y_hat))
        self.unit_test.assertTrue(np.isclose(cs, 1), msg=f'fail cosine similarity check:{cs}')

class TwoConv2DCollapsingTest(BaseConv2DCollapsingTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    class Conv2DCollapsingNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(16, 256, kernel_size=(5,5), stride=(1,1), padding=(1,1))
            self.conv2 = nn.Conv2d(256, 4, kernel_size=(1,1), stride=(1,1))

        def forward(self, x):
            x = self.conv1(x)
            y = self.conv2(x)
            return y

    def create_networks(self):
        return self.Conv2DCollapsingNet()

class ThreeConv2DCollapsingTest(BaseConv2DCollapsingTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    class Conv2DCollapsingNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(16, 128, kernel_size=(3,3), stride=(1,1), bias=False)
            self.conv2 = nn.Conv2d(128, 64, kernel_size=(1,1), stride=(1,1), bias=False)
            self.conv3 = nn.Conv2d(64, 16, kernel_size=(1,1), stride=(1,1), bias=False)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            y = self.relu(x)
            return y

    def create_networks(self):
        return self.Conv2DCollapsingNet()

class FourConv2DCollapsingTest(BaseConv2DCollapsingTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    class Conv2DCollapsingNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(16, 128, kernel_size=(3, 3), stride=(1, 1))
            self.conv2 = nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
            self.conv3 = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
            self.conv4 = nn.Conv2d(64, 4, kernel_size=(1, 1), stride=(1, 1))
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            y = self.relu(x)
            return y

    def create_networks(self):
        return self.Conv2DCollapsingNet()

class SixConv2DCollapsingTest(BaseConv2DCollapsingTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    class Conv2DCollapsingNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(16, 32, kernel_size=(5,5), stride=(1,1))
            self.conv2 = nn.Conv2d(32, 4, kernel_size=(1,1), stride=(1,1), bias=False)
            self.conv3 = nn.Conv2d(4, 128, kernel_size=(1,1), stride=(1,1))
            self.conv4 = nn.Conv2d(128, 16, kernel_size=(3,3), stride=(1,1))
            self.conv5 = nn.Conv2d(16, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.conv6 = nn.Conv2d(64, 8, kernel_size=(1, 1), stride=(1, 1))
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.relu(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.conv5(x)
            x = self.conv6(x)
            y = self.relu(x)
            return y

    def create_networks(self):
        return self.Conv2DCollapsingNet()

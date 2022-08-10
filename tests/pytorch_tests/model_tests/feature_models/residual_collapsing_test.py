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
import torch
import torch.nn as nn
import numpy as np
import model_compression_toolkit as mct
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor, torch_tensor_to_numpy
from tests.pytorch_tests.model_tests.base_pytorch_feature_test import BasePytorchFeatureNetworkTest
from tests.common_tests.helpers.generate_test_tp_model import generate_test_tp_model
from model_compression_toolkit.core.tpc_models.default_tpc.latest import generate_pytorch_tpc
from tests.common_tests.helpers.tensors_compare import cosine_similarity

tp = mct.target_platform


class BaseResidualCollapsingTest(BasePytorchFeatureNetworkTest):

    def __init__(self, unit_test):
        super().__init__(unit_test=unit_test, input_shape=(3,16,16))

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
        for layer in quantized_model.children():
            self.unit_test.assertFalse(type(layer) == torch.add, msg=f'fail: add residual is still in the model')
        cs = cosine_similarity(torch_tensor_to_numpy(y), torch_tensor_to_numpy(y_hat))
        self.unit_test.assertTrue(np.isclose(cs, 1), msg=f'fail: cosine similarity check:{cs}')

class ResidualCollapsingTest1(BaseResidualCollapsingTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    class ResidualCollapsingNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 3, kernel_size=(3,3), padding='same')
            self.relu = nn.ReLU()

        def forward(self, x):
            y = self.conv1(x)
            y = torch.add(x,y)
            y = self.relu(y)
            return y

    def create_networks(self):
        return self.ResidualCollapsingNet()

class ResidualCollapsingTest2(BaseResidualCollapsingTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    class ResidualCollapsingNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 3, kernel_size=(3,4), padding='same')
            self.conv2 = nn.Conv2d(3, 3, kernel_size=(2,2), padding='same')
            self.conv3 = nn.Conv2d(3, 3, kernel_size=(1,3), padding='same', bias=False)
            self.relu = nn.ReLU()

        def forward(self, x):
            x1 = self.conv1(x)
            x2 = torch.add(x1, x)
            x3 = self.conv2(x2)
            x4 = torch.add(x3, x2)
            x4 = self.relu(x4)
            x5 = self.conv3(x4)
            y = torch.add(x4, x5)
            y = self.relu(y)
            return y

    def create_networks(self):
        return self.ResidualCollapsingNet()

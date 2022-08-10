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
import model_compression_toolkit as mct
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor
from tests.pytorch_tests.model_tests.base_pytorch_feature_test import BasePytorchFeatureNetworkTest
from tests.common_tests.helpers.generate_test_tp_model import generate_test_tp_model
from model_compression_toolkit.core.tpc_models.default_tpc.latest import generate_pytorch_tpc


tp = mct.target_platform


class BasePermuteSubstitutionTest(BasePytorchFeatureNetworkTest):

    def __init__(self, unit_test):
        super().__init__(unit_test=unit_test)

    def get_tpc(self):
        tp = generate_test_tp_model({'weights_n_bits': 32,
                                     'activation_n_bits': 32,
                                     'enable_weights_quantization': False,
                                     'enable_activation_quantization': False})
        return generate_pytorch_tpc(name="permute_substitution_test", tp_model=tp)

    def get_quantization_config(self):
        return mct.QuantizationConfig(mct.QuantizationErrorMethod.NOCLIPPING, mct.QuantizationErrorMethod.NOCLIPPING,
                                      False, False, True)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        in_torch_tensor = to_torch_tensor(input_x[0])
        y = float_model(in_torch_tensor)
        y_hat = quantized_model(in_torch_tensor)
        self.unit_test.assertTrue(y.shape == y_hat.shape, msg=f'out shape is not as expected!')

class PermuteSubstitutionTest(BasePermuteSubstitutionTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    class PermuteNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=(3,3))

        def forward(self, x):
            x = self.conv1(x)
            x = torch.permute(x, (3,0,1,2))
            x = torch.permute(x, (1,2,3,0))
            x = x.permute(2,0,1,3)
            x = x.permute(1,3,0,2)
            return x

    def create_networks(self):
        return self.PermuteNet()

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

from model_compression_toolkit.core.tpc_models.default_tpc.latest import get_tp_model
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor
from tests.pytorch_tests.tpc_pytorch import get_pytorch_test_tpc_dict
from tests.pytorch_tests.model_tests.base_pytorch_test import BasePytorchTest
import model_compression_toolkit as mct
import numpy as np

"""
This test checks the shift negative activation feature.
"""


class ShiftNegaviteActivationNet(torch.nn.Module):
    def __init__(self):
        super(ShiftNegaviteActivationNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, kernel_size=(5,6), stride=2)
        self.conv2 = torch.nn.Conv2d(4, 5, kernel_size=(8,7), stride=2, bias=False)
        self.activation = torch.nn.Hardswish()

    def forward(self, inp):
        x0 = self.conv1(inp)
        x1 = self.activation(x0)
        x2 = self.conv2(x1)
        x3 = self.activation(x2)
        return inp, x0, x1, x2, x3


class ShiftNegaviteActivationNetTest(BasePytorchTest):
    """
    This test checks the shift negative activation feature.
    """
    def __init__(self, unit_test, float_reconstruction_error=1e-6):
        super().__init__(unit_test, float_reconstruction_error)

    @staticmethod
    def generate_inputs(input_shapes):
        i = to_torch_tensor([torch.randn(*in_shape) for in_shape in input_shapes])
        i[0][0, 0, 0, 0] = 10
        i[0][0, 0, 0, 1] = -10
        return i

    def get_tpc(self):
        return get_pytorch_test_tpc_dict(tp_model=get_tp_model(),
                                         test_name='all_8bit',
                                         ftp_name='sn_pytorch_test')

    def get_quantization_configs(self):
        return {
            'all_8bit': mct.QuantizationConfig(mct.QuantizationErrorMethod.NOCLIPPING,
                                               mct.QuantizationErrorMethod.NOCLIPPING,
                                               shift_negative_activation_correction=True,
                                               shift_negative_ratio=np.inf),
        }

    def create_inputs_shape(self):
        return [[self.val_batch_size, 3, 224, 224]]

    def create_feature_network(self, input_shape):
        return ShiftNegaviteActivationNet()
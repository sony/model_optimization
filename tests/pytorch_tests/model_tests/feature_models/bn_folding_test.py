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
import numpy as np
from model_compression_toolkit.core.pytorch.utils import set_model, to_torch_tensor, \
    torch_tensor_to_numpy
from tests.pytorch_tests.model_tests.base_pytorch_test import BasePytorchTest

"""
This test checks the BatchNorm folding feature, plus adding a residual connection.
"""
class BNFoldingNet(torch.nn.Module):
    def __init__(self, test_layer, fold_applied):
        super(BNFoldingNet, self).__init__()
        self.conv1 = test_layer
        self.fold_applied = fold_applied
        self.bn = torch.nn.BatchNorm2d(test_layer.out_channels)

    def forward(self, inp):
        x1 = self.conv1(inp)
        x = self.bn(x1)
        x = torch.relu(x)
        if not self.fold_applied:
            x = x + x1
        return x


class BNFoldingNetTest(BasePytorchTest):
    """
    This test checks the BatchNorm folding feature, plus adding a residual connection.
    """
    def __init__(self, unit_test, test_layer, fold_applied=True, float_reconstruction_error=1e-6):
        super().__init__(unit_test, float_reconstruction_error)
        self.test_layer = test_layer
        self.fold_applied = fold_applied

    def create_feature_network(self, input_shape):
        return BNFoldingNet(self.test_layer, self.fold_applied)

    def get_tpc(self):
        return {'no_quantization': super().get_tpc()['no_quantization']}

    def get_core_configs(self):
        return {'no_quantization': super().get_core_configs()['no_quantization']}

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        set_model(float_model)
        quant_model = quantized_models['no_quantization']
        set_model(quant_model)
        out_float = torch_tensor_to_numpy(float_model(*input_x))
        out_quant = torch_tensor_to_numpy(quant_model(*input_x))

        is_bn_in_model = torch.nn.BatchNorm2d in [type(module) for name, module in quant_model.named_modules()]
        self.unit_test.assertTrue(self.fold_applied is not is_bn_in_model)
        self.unit_test.assertTrue(np.isclose(out_quant, out_float, atol=1e-6, rtol=1e-4).all())


class BNForwardFoldingNet(torch.nn.Module):
    def __init__(self, test_layer, add_bn=False):
        super(BNForwardFoldingNet, self).__init__()
        self.bn = torch.nn.BatchNorm2d(3)
        torch.nn.init.uniform_(self.bn.weight, 0.02, 1.05)
        torch.nn.init.uniform_(self.bn.bias, -1.2, 1.05)
        torch.nn.init.uniform_(self.bn.running_var, 0.02, 1.05)
        torch.nn.init.uniform_(self.bn.running_mean, -1.2, 1.05)
        self.conv = test_layer
        if add_bn:
            self.bn2 = torch.nn.BatchNorm2d(test_layer.out_channels)
            torch.nn.init.uniform_(self.bn2.weight, 0.02, 1.05)
            torch.nn.init.uniform_(self.bn2.bias, -1.2, 1.05)
            torch.nn.init.uniform_(self.bn2.running_var, 0.02, 1.05)
            torch.nn.init.uniform_(self.bn2.running_mean, -1.2, 1.05)
        else:
            self.bn2 = None

    def forward(self, inp):
        x = self.bn(inp)
        x = self.conv(x)
        if self.bn2 is not None:
            x = self.bn2(x)
        x = torch.tanh(x)
        return x


class BNForwardFoldingNetTest(BasePytorchTest):
    """
    This test checks the BatchNorm forward folding feature. When fold_applied is False
    test that the BN isn't folded
    """
    def __init__(self, unit_test, test_layer, fold_applied=True, add_bn=False):
        super().__init__(unit_test, float_reconstruction_error=1e-6)
        self.test_layer = test_layer
        self.fold_applied = fold_applied
        self.add_bn = add_bn

    def create_feature_network(self, input_shape):
        return BNForwardFoldingNet(self.test_layer, self.add_bn)

    def get_tpc(self):
        return {'no_quantization': super().get_tpc()['no_quantization']}

    def get_core_configs(self):
        return {'no_quantization': super().get_core_configs()['no_quantization']}

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        set_model(float_model)
        quant_model = quantized_models['no_quantization']
        set_model(quant_model)
        out_float = torch_tensor_to_numpy(float_model(*input_x))
        out_quant = torch_tensor_to_numpy(quant_model(*input_x))

        is_bn_in_model = torch.nn.BatchNorm2d in [type(module) for name, module in quant_model.named_modules()]
        self.unit_test.assertTrue(self.fold_applied is not is_bn_in_model)
        self.unit_test.assertTrue(np.isclose(out_quant, out_float, atol=1e-6, rtol=1e-4).all())

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

from mct_quantizers import PytorchQuantizationWrapper
from tests.pytorch_tests.model_tests.base_pytorch_test import BasePytorchTest
import model_compression_toolkit as mct

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReuseLayerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv2(x)
        return x


class ReuseFunctionalLayerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_weight = nn.Parameter(torch.randn(3, 3, 2, 2))
        self.conv1_bias = nn.Parameter(torch.randn(3))
        self.conv2_weight = nn.Parameter(torch.randn(3, 3, 2, 2))
        self.conv2_bias = nn.Parameter(torch.randn(3))

    def forward(self, x):
        x = F.conv2d(x, self.conv1_weight, self.conv1_bias, stride=1, groups=1)
        x = F.conv2d(x, self.conv1_weight, self.conv1_bias, stride=1, groups=1)
        x = F.conv2d(x, self.conv2_weight, self.conv2_bias, stride=1, groups=1)
        x = F.conv2d(x, self.conv2_weight, self.conv2_bias, stride=1, groups=1)

        # Not reused layer
        x = F.conv2d(x, self.conv1_weight, self.conv2_bias, stride=1, groups=1)
        x = F.conv2d(x, self.conv2_weight, self.conv1_bias, stride=1, groups=1)
        return x


class CombinedReuseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.func_conv_weight = nn.Parameter(torch.randn(3, 3, 3, 3))
        self.func_conv_bias = nn.Parameter(torch.randn(3))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1(x)
        x = self.conv1(x)
        x = F.conv2d(x, self.func_conv_weight, self.func_conv_bias, stride=1, padding=1)
        x = F.conv2d(x, self.func_conv_weight, self.func_conv_bias, stride=1, padding=1)
        return x


class ReuseLayerNetTest(BasePytorchTest):
    def create_feature_network(self, input_shape):
        return ReuseLayerNet()

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        model = quantized_models['all_4bit']
        self.unit_test.assertEqual(
            [p.data_ptr() for p in model.conv1.parameters()],
            [p.data_ptr() for p in model.conv1_1.parameters()],
            f"Shared parameters between reused conv1 layers should have identical memory addresses"
        )
        self.unit_test.assertEqual(
            [p.data_ptr() for p in model.conv2.parameters()],
            [p.data_ptr() for p in model.conv2_1.parameters()],
            f"Shared parameters between reused conv2 layers should have identical memory addresses"
        )
        self.unit_test.assertEqual(
            [p.data_ptr() for p in model.conv2.parameters()],
            [p.data_ptr() for p in model.conv2_2.parameters()],
            f"Shared parameters between reused conv2 layers should have identical memory addresses"
        )
        self.unit_test.assertNotEqual(
            [p.data_ptr() for p in model.conv1.parameters()],
            [p.data_ptr() for p in model.conv2.parameters()],
            f"Parameters between conv1 and conv2 layers should have different memory addresses"
        )

        layer_calls = count_layer_calls(model, input_x)
        self.unit_test.assertEqual(layer_calls['conv1'], 2, "conv1 should be called twice")
        self.unit_test.assertEqual(layer_calls['conv2'], 3, "conv2 should be called three times")


class ReuseFunctionalLayerNetTest(BasePytorchTest):
    def create_feature_network(self, input_shape):
        return ReuseFunctionalLayerNet()

    def get_quantization_config(self):
        return mct.core.QuantizationConfig(linear_collapsing=False)

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        model = quantized_models['all_4bit']
        self.unit_test.assertEqual(
            [p.data_ptr() for p in model.conv2d.parameters()],
            [p.data_ptr() for p in model.conv2d_1.parameters()],
            f"Shared parameters between reused conv2d layers should have identical memory addresses"
        )
        self.unit_test.assertEqual(
            [p.data_ptr() for p in model.conv2d_2.parameters()],
            [p.data_ptr() for p in model.conv2d_3.parameters()],
            f"Shared parameters between reused conv2d_2 layers should have identical memory addresses"
        )
        self.unit_test.assertNotEqual(
            [p.data_ptr() for p in model.conv2d.parameters()],
            [p.data_ptr() for p in model.conv2d_2.parameters()],
            f"Parameters between conv2d and conv2d_2 layers should have different memory addresses"
        )

        layer_calls = count_layer_calls(model, input_x)
        self.unit_test.assertEqual(layer_calls['conv2d'], 2, "conv2d should be called twice")
        self.unit_test.assertEqual(layer_calls['conv2d_2'], 2, "conv2d_2 should be called twice")

        # Check the two layers that are not reused:
        self.unit_test.assertEqual(layer_calls['conv2d_4'], 1, "conv2d_4 should be called twice")
        self.unit_test.assertEqual(layer_calls['conv2d_5'], 1, "conv2d_5 should be called twice")

        # Check that conv2d_4 is not considered reused with all other conv layers
        self.unit_test.assertNotEqual(
            [p.data_ptr() for p in model.conv2d_4.parameters()],
            [p.data_ptr() for p in model.conv2d.parameters()],
            f"Parameters between conv2d_4 and conv2d layers should have different memory addresses"
        )
        self.unit_test.assertNotEqual(
            [p.data_ptr() for p in model.conv2d_4.parameters()],
            [p.data_ptr() for p in model.conv2d_2.parameters()],
            f"Parameters between conv2d_4 and conv2d_2 layers should have different memory addresses"
        )
        self.unit_test.assertNotEqual(
            [p.data_ptr() for p in model.conv2d_4.parameters()],
            [p.data_ptr() for p in model.conv2d_5.parameters()],
            f"Parameters between conv2d_4 and conv2d_5 layers should have different memory addresses"
        )

        # Check that conv2d_5 is not considered reused with all other conv layers
        self.unit_test.assertNotEqual(
            [p.data_ptr() for p in model.conv2d_5.parameters()],
            [p.data_ptr() for p in model.conv2d.parameters()],
            f"Parameters between conv2d_5 and conv2d layers should have different memory addresses"
        )
        self.unit_test.assertNotEqual(
            [p.data_ptr() for p in model.conv2d_5.parameters()],
            [p.data_ptr() for p in model.conv2d_2.parameters()],
            f"Parameters between conv2d_5 and conv2d_2 layers should have different memory addresses"
        )
        self.unit_test.assertNotEqual(
            [p.data_ptr() for p in model.conv2d_5.parameters()],
            [p.data_ptr() for p in model.conv2d_4.parameters()],
            f"Parameters between conv2d_5 and conv2d_4 layers should have different memory addresses"
        )


class ReuseModuleAndFunctionalLayersTest(BasePytorchTest):
    def create_feature_network(self, input_shape):
        return CombinedReuseNet()

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        model = quantized_models['all_4bit']
        self.unit_test.assertEqual(
            [p.data_ptr() for p in model.conv1.parameters()],
            [p.data_ptr() for p in model.conv1_1.parameters()],
            f"Shared parameters between reused conv1 layers should have identical memory addresses"
        )
        self.unit_test.assertEqual(
            [p.data_ptr() for p in model.conv2d.parameters()],
            [p.data_ptr() for p in model.conv2d_1.parameters()],
            f"Shared parameters between reused conv2d layers should have identical memory addresses"
        )
        self.unit_test.assertNotEqual(
            [p.data_ptr() for p in model.conv1.parameters()],
            [p.data_ptr() for p in model.conv2d.parameters()],
            f"Parameters between conv1 and conv2d layers should have different memory addresses"
        )

        layer_calls = count_layer_calls(model, input_x)
        self.unit_test.assertEqual(layer_calls['conv1'], 3, "conv1 should be called three times")
        self.unit_test.assertEqual(layer_calls['conv2d'], 2, "conv2d should be called twice")


def count_layer_calls(model, input_x):
    layer_calls = {}

    def hook_fn(module, input, output):
        layer_name = [name for name, layer in model.named_modules() if layer is module][0]
        layer_calls[layer_name] = layer_calls.get(layer_name, 0) + 1

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, PytorchQuantizationWrapper):
            hooks.append(module.register_forward_hook(hook_fn))

    _ = model(input_x)

    for hook in hooks:
        hook.remove()

    return layer_calls

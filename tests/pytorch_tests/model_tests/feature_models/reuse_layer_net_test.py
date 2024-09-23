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

from mct_quantizers import PytorchQuantizationWrapper
from tests.pytorch_tests.model_tests.base_pytorch_test import BasePytorchTest
import torch.nn.functional as F


#
# class ReuseLayerNet(torch.nn.Module):
#     def __init__(self):
#         super(ReuseLayerNet, self).__init__()
#         self.conv1 = torch.nn.Conv2d(3, 3, kernel_size=1, stride=1)
#         self.conv2 = torch.nn.Conv2d(3, 3, kernel_size=1, stride=1)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv2(x)
#         return x
#
#
# class ReuseLayerNetTest(BasePytorchTest):
#     """
#     This test checks:
#     The reuse of a layer in a model.
#     """
#     def __init__(self, unit_test):
#         super().__init__(unit_test)
#
#     def create_inputs_shape(self):
#         return [[self.val_batch_size, 3, 32, 32]]
#
#     def create_feature_network(self, input_shape):
#         model = ReuseLayerNet()
#         return model
#
#     def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
#
#         quant_model = quantized_models['all_4bit']
#
#         #########################################################################################
#
#         # Verify that the shared parameters have identical memory addresses
#         self.unit_test.assertEqual([p.data_ptr() for p in quant_model.conv1.parameters()],
#                                    [p.data_ptr() for p in quant_model.conv1_1.parameters()],
#                                    f"Shared parameters between reused layers should have identical memory addresses")
#         self.unit_test.assertEqual([p.data_ptr() for p in quant_model.conv2.parameters()],
#                                    [p.data_ptr() for p in quant_model.conv2_1.parameters()],
#                                    f"Shared parameters between reused layers should have identical memory addresses")
#         self.unit_test.assertEqual([p.data_ptr() for p in quant_model.conv2.parameters()],
#                                    [p.data_ptr() for p in quant_model.conv2_2.parameters()],
#                                    f"Shared parameters between reused layers should have identical memory addresses")
#         self.unit_test.assertNotEqual([p.data_ptr() for p in quant_model.conv1.parameters()],
#                                       [p.data_ptr() for p in quant_model.conv2.parameters()],
#                                       f"Parameters between different layers should have different memory addresses")
#
#         #########################################################################################
#
#         # Verify that 'conv1' is called twice (thus reused) and 'conv2' is called three times
#         layer_calls = {}
#         def hook_fn(module, input, output):
#             layer_name = [name for name, layer in quant_model.named_modules() if layer is module][0]
#             if layer_name not in layer_calls:
#                 layer_calls[layer_name] = 0
#             layer_calls[layer_name] += 1
#
#         # Register hooks
#         hooks = []
#         for name, module in quant_model.named_modules():
#             if isinstance(module, PytorchQuantizationWrapper):
#                 hooks.append(module.register_forward_hook(hook_fn))
#         _ = quant_model(input_x)
#         for hook in hooks:
#             hook.remove()
#
#         self.unit_test.assertEqual(layer_calls['conv1'], 2, "conv1 should be called twice")
#         self.unit_test.assertEqual(layer_calls['conv2'], 3, "conv2 should be called three times")
#
#         #########################################################################################
#
#
#
#
#
#
# class ReuseFunctionalLayerNet(torch.nn.Module):
#     def __init__(self):
#         super(ReuseFunctionalLayerNet, self).__init__()
#         # Define the weight and bias manually for the functional conv2d
#         self.conv1_weight = torch.nn.Parameter(torch.randn(3, 3, 2, 2))
#         self.conv1_bias = torch.nn.Parameter(torch.randn(3))
#         self.conv2_weight = torch.nn.Parameter(torch.randn(3, 3, 2, 2))
#         self.conv2_bias = torch.nn.Parameter(torch.randn(3))
#
#     def forward(self, x):
#         x = F.conv2d(x, self.conv1_weight, self.conv1_bias, stride=1, groups=1)
#         x = F.conv2d(x, self.conv1_weight, self.conv1_bias, stride=1, groups=1)
#         x = F.conv2d(x, self.conv2_weight, self.conv2_bias, stride=1, groups=1)
#         x = F.conv2d(x, self.conv2_weight, self.conv2_bias, stride=1, groups=1)
#         return x
#
#
# class ReuseFunctionalLayerNetTest(BasePytorchTest):
#     """
#     This test checks:
#     The reuse of a layer in a model.
#     """
#     def __init__(self, unit_test):
#         super().__init__(unit_test)
#
#     def create_inputs_shape(self):
#         return [[self.val_batch_size, 3, 32, 32]]
#
#
#     def create_feature_network(self, input_shape):
#         model = ReuseFunctionalLayerNet()
#         return model
#
#     def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
#
#         quant_model = quantized_models['all_4bit']
#         # Verify that the shared parameters have identical memory addresses
#         self.unit_test.assertEqual([p.data_ptr() for p in quant_model.conv2d.parameters()],
#                                    [p.data_ptr() for p in quant_model.conv2d_1.parameters()],
#                                    f"Shared parameters between reused layers should have identical memory addresses")
#         self.unit_test.assertEqual([p.data_ptr() for p in quant_model.conv2d_2.parameters()],
#                                    [p.data_ptr() for p in quant_model.conv2d_3.parameters()],
#                                    f"Shared parameters between reused layers should have identical memory addresses")
#         self.unit_test.assertNotEqual([p.data_ptr() for p in quant_model.conv2d.parameters()],
#                                       [p.data_ptr() for p in quant_model.conv2d_2.parameters()],
#                                       f"Parameters between different layers should have different memory addresses")
#         # Count layer calls
#         layer_calls = {}
#
#         def hook_fn(module, input, output):
#             layer_name = [name for name, layer in quant_model.named_modules() if layer is module][0]
#             if layer_name not in layer_calls:
#                 layer_calls[layer_name] = 0
#             layer_calls[layer_name] += 1
#
#         # Register hooks
#         hooks = []
#         for name, module in quant_model.named_modules():
#             if isinstance(module, PytorchQuantizationWrapper):
#                 hooks.append(module.register_forward_hook(hook_fn))
#
#         _ = quant_model(input_x)
#
#         for hook in hooks:
#             hook.remove()
#
#         # Verify call counts
#         self.unit_test.assertEqual(layer_calls['conv2d'], 2, "conv2d should be called twice")
#         self.unit_test.assertEqual(layer_calls['conv2d_2'], 2, "conv2d_2 should be called twice")
#
#
#
# class CombinedReuseNet(torch.nn.Module):
#     def __init__(self):
#         super(CombinedReuseNet, self).__init__()
#         self.conv1 = torch.nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
#
#         # Functional layer parameters
#         self.func_conv_weight = torch.nn.Parameter(torch.randn(3, 3, 3, 3))
#         self.func_conv_bias = torch.nn.Parameter(torch.randn(3))
#
#     def forward(self, x):
#         # Reused module layers
#         x = self.conv1(x)
#         x = self.conv1(x)
#         x = self.conv1(x)
#
#         # Reused functional layers
#         x = F.conv2d(x, self.func_conv_weight, self.func_conv_bias, stride=1, padding=1)
#         x = F.conv2d(x, self.func_conv_weight, self.func_conv_bias, stride=1, padding=1)  # Reuse functional conv
#
#         return x
#
#
# class ReuseModuleAndFunctionalLayersTest(BasePytorchTest):
#     """
#     This test checks:
#     1. The reuse of module layers in a model.
#     2. The reuse of functional layers in the same model.
#     """
#
#     def __init__(self, unit_test):
#         super().__init__(unit_test)
#
#     def create_inputs_shape(self):
#         return [[self.val_batch_size, 3, 32, 32]]
#
#     def create_feature_network(self, input_shape):
#         return CombinedReuseNet()
#
#     def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
#         quant_model = quantized_models['all_4bit']
#
#         # Verify that the shared parameters for module layers have identical memory addresses
#         self.unit_test.assertEqual([p.data_ptr() for p in quant_model.conv1.parameters()],
#                                    [p.data_ptr() for p in quant_model.conv1_1.parameters()],
#                                    "Shared parameters between reused module layers should have identical memory addresses")
#
#         # Verify that the shared parameters for functional layers have identical memory addresses
#         self.unit_test.assertEqual([p.data_ptr() for p in quant_model.conv2d.parameters()],
#                                    [p.data_ptr() for p in quant_model.conv2d_1.parameters()],
#                                    "Shared parameters between reused functional layers should have identical memory addresses")
#
#         # Verify that parameters between different layers have different memory addresses
#         self.unit_test.assertNotEqual([p.data_ptr() for p in quant_model.conv1.parameters()],
#                                       [p.data_ptr() for p in quant_model.conv2d.parameters()],
#                                       "Parameters between different module layers should have different memory addresses")
#
#         # Count layer calls
#         layer_calls = {}
#
#         def hook_fn(module, input, output):
#             layer_name = [name for name, layer in quant_model.named_modules() if layer is module][0]
#             if layer_name not in layer_calls:
#                 layer_calls[layer_name] = 0
#             layer_calls[layer_name] += 1
#
#         # Register hooks
#         hooks = []
#         for name, module in quant_model.named_modules():
#             if isinstance(module, PytorchQuantizationWrapper):
#                 hooks.append(module.register_forward_hook(hook_fn))
#
#         _ = quant_model(input_x)
#
#         for hook in hooks:
#             hook.remove()
#
#         # Verify call counts
#         self.unit_test.assertEqual(layer_calls['conv1'], 3, "conv1 should be called twice")
#         self.unit_test.assertEqual(layer_calls['conv2d'], 2, "conv2d should be called twice")


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any

class BaseReuseLayerNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward method")

class ReuseLayerNet(BaseReuseLayerNet):
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

class ReuseFunctionalLayerNet(BaseReuseLayerNet):
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
        return x

class CombinedReuseNet(BaseReuseLayerNet):
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

class ReuseLayerTestBase(BasePytorchTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_inputs_shape(self):
        return [[self.val_batch_size, 3, 32, 32]]

    def create_feature_network(self, input_shape):
        return self.get_model()

    def get_model(self):
        raise NotImplementedError("Subclasses must implement get_model method")

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        quant_model = quantized_models['all_4bit']
        self.verify_shared_parameters(quant_model)
        self.verify_layer_calls(quant_model, input_x)

    def verify_shared_parameters(self, model):
        raise NotImplementedError("Subclasses must implement verify_shared_parameters method")

    def verify_layer_calls(self, model, input_x):
        layer_calls = self.count_layer_calls(model, input_x)
        self.check_layer_call_counts(layer_calls)

    def count_layer_calls(self, model, input_x):
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

    def check_layer_call_counts(self, layer_calls):
        raise NotImplementedError("Subclasses must implement check_layer_call_counts method")

    def assert_shared_parameters(self, layer1, layer2, layer_name):
        self.unit_test.assertEqual(
            [p.data_ptr() for p in layer1.parameters()],
            [p.data_ptr() for p in layer2.parameters()],
            f"Shared parameters between reused {layer_name} layers should have identical memory addresses"
        )

    def assert_different_parameters(self, layer1, layer2, layer_names):
        self.unit_test.assertNotEqual(
            [p.data_ptr() for p in layer1.parameters()],
            [p.data_ptr() for p in layer2.parameters()],
            f"Parameters between {layer_names} layers should have different memory addresses"
        )


class ReuseLayerNetTest(ReuseLayerTestBase):
    def get_model(self):
        return ReuseLayerNet()

    def verify_shared_parameters(self, model):
        self.assert_shared_parameters(model.conv1, model.conv1_1, "conv1")
        self.assert_shared_parameters(model.conv2, model.conv2_1, "conv2")
        self.assert_shared_parameters(model.conv2, model.conv2_2, "conv2")
        self.assert_different_parameters(model.conv1, model.conv2, "conv1 and conv2")

    def check_layer_call_counts(self, layer_calls):
        self.unit_test.assertEqual(layer_calls['conv1'], 2, "conv1 should be called twice")
        self.unit_test.assertEqual(layer_calls['conv2'], 3, "conv2 should be called three times")

class ReuseFunctionalLayerNetTest(ReuseLayerTestBase):
    def get_model(self):
        return ReuseFunctionalLayerNet()

    def verify_shared_parameters(self, model):
        self.assert_shared_parameters(model.conv2d, model.conv2d_1, "conv2d")
        self.assert_shared_parameters(model.conv2d_2, model.conv2d_3, "conv2d_2")
        self.assert_different_parameters(model.conv2d, model.conv2d_2, "conv2d and conv2d_2")

    def check_layer_call_counts(self, layer_calls):
        self.unit_test.assertEqual(layer_calls['conv2d'], 2, "conv2d should be called twice")
        self.unit_test.assertEqual(layer_calls['conv2d_2'], 2, "conv2d_2 should be called twice")

class ReuseModuleAndFunctionalLayersTest(ReuseLayerTestBase):
    def get_model(self):
        return CombinedReuseNet()

    def verify_shared_parameters(self, model):
        self.assert_shared_parameters(model.conv1, model.conv1_1, "conv1")
        self.assert_shared_parameters(model.conv2d, model.conv2d_1, "conv2d")
        self.assert_different_parameters(model.conv1, model.conv2d, "conv1 and conv2d")

    def check_layer_call_counts(self, layer_calls):
        self.unit_test.assertEqual(layer_calls['conv1'], 3, "conv1 should be called three times")
        self.unit_test.assertEqual(layer_calls['conv2d'], 2, "conv2d should be called twice")


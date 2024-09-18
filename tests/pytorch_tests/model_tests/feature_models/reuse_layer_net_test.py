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

"""
This test checks:
The reuse of a layer in a model.
"""

class ReuseLayerNet(torch.nn.Module):
    def __init__(self):
        super(ReuseLayerNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, kernel_size=1, stride=1)
        self.conv2 = torch.nn.Conv2d(3, 3, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv2(x)
        return x


class ReuseLayerNetTest(BasePytorchTest):
    """
    This test checks:
    The reuse of a layer in a model.
    """
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_inputs_shape(self):
        return [[self.val_batch_size, 3, 32, 32]]

    def create_feature_network(self, input_shape):
        model = ReuseLayerNet()
        return model

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):

        quant_model = quantized_models['all_4bit']

        #########################################################################################

        # Verify that the shared parameters have identical memory addresses
        self.unit_test.assertEqual([p.data_ptr() for p in quant_model.conv1.parameters()],
                                   [p.data_ptr() for p in quant_model.conv1_1.parameters()],
                                   f"Shared parameters between reused layers should have identical memory addresses")
        self.unit_test.assertEqual([p.data_ptr() for p in quant_model.conv2.parameters()],
                                   [p.data_ptr() for p in quant_model.conv2_1.parameters()],
                                   f"Shared parameters between reused layers should have identical memory addresses")
        self.unit_test.assertEqual([p.data_ptr() for p in quant_model.conv2.parameters()],
                                   [p.data_ptr() for p in quant_model.conv2_2.parameters()],
                                   f"Shared parameters between reused layers should have identical memory addresses")
        self.unit_test.assertNotEqual([p.data_ptr() for p in quant_model.conv1.parameters()],
                                      [p.data_ptr() for p in quant_model.conv2.parameters()],
                                      f"Parameters between different layers should have different memory addresses")

        #########################################################################################

        # Verify that 'conv1' is called twice (thus reused) and 'conv2' is called three times
        layer_calls = {}
        def hook_fn(module, input, output):
            layer_name = [name for name, layer in quant_model.named_modules() if layer is module][0]
            if layer_name not in layer_calls:
                layer_calls[layer_name] = 0
            layer_calls[layer_name] += 1

        # Register hooks
        hooks = []
        for name, module in quant_model.named_modules():
            if isinstance(module, PytorchQuantizationWrapper):
                hooks.append(module.register_forward_hook(hook_fn))
        _ = quant_model(input_x)
        for hook in hooks:
            hook.remove()

        self.unit_test.assertEqual(layer_calls['conv1'], 2, "conv1 should be called twice")
        self.unit_test.assertEqual(layer_calls['conv2'], 3, "conv2 should be called three times")

        #########################################################################################

# Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
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

import numpy as np
import torch
from torch.nn import GELU, Hardswish, AdaptiveAvgPool2d, ZeroPad2d, Linear, Conv2d

import model_compression_toolkit as mct
from tests.pytorch_tests.model_tests.base_pytorch_feature_test import BasePytorchFeatureNetworkTest

"""
This test checks the Activation Bias Correction feature.
"""


class ActivationBiasCorrectionNet(torch.nn.Module):
    """
    This is the network to test the Activation Bias Correction feature.
    """

    def __init__(self,
                 linear_layer,
                 bypass_layers):
        super(ActivationBiasCorrectionNet, self).__init__()
        self.activation_layer = GELU()
        self.linear_layer = linear_layer
        self.bypass_layers = torch.nn.ModuleList(bypass_layers)

    def forward(self, x):
        x = self.activation_layer(x)

        for bypass_layer in self.bypass_layers:
            x = bypass_layer(x)
        x = self.linear_layer(x)
        return x


class ActivationBiasCorrectionPadNet(torch.nn.Module):
    """
    This is the network to test the Activation Bias Correction feature with pooling/padding layers as a bypass layers.
    """

    def __init__(self):
        super(ActivationBiasCorrectionPadNet, self).__init__()
        self.activation_layer = Hardswish()
        self.pooling_layer = AdaptiveAvgPool2d(output_size=6)
        self.padding_layer = ZeroPad2d(padding=2)
        self.linear_layer = Linear(10, 10)

    def forward(self, x):
        x = self.activation_layer(x)
        x = self.pooling_layer(x)
        x = self.padding_layer(x)
        x = self.linear_layer(x)
        return x


class ActivationBiasCorrectionReshapeNet(torch.nn.Module):
    """
    This is the network to test the Activation Bias Correction feature with reshape layers as a bypass layers.
    """

    def __init__(self):
        super(ActivationBiasCorrectionReshapeNet, self).__init__()
        self.activation_layer = GELU()
        self.linear_layer = Conv2d(in_channels=8, out_channels=1, kernel_size=1)

    def forward(self, x):
        x = self.activation_layer(x)
        x = x.flatten()
        x = x.reshape(8, 2, -1)
        x = self.linear_layer(x)
        return x


class BaseActivationBiasCorrectionTest(BasePytorchFeatureNetworkTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_quantization_config(self):
        return mct.core.QuantizationConfig(weights_bias_correction=False,
                                           weights_second_moment_correction=False,
                                           activation_bias_correction=True)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        bias = float_model.linear_layer.bias.cpu().detach().numpy()
        bias_after_activation_bias_correction = quantized_model.linear_layer.layer.bias.cpu().detach().numpy()
        self.unit_test.assertFalse(np.array_equal(bias, bias_after_activation_bias_correction),
                                   msg=f"Error in activation bias correction: expected a change in the bias value.")


class BaseActivationBiasCorrectionNetTest(BaseActivationBiasCorrectionTest):
    def __init__(self, unit_test, linear_layer, bypass_layers):
        super().__init__(unit_test)
        self.linear_layer = linear_layer
        self.bypass_layers = bypass_layers

    def create_networks(self):
        return ActivationBiasCorrectionNet(linear_layer=self.linear_layer,
                                           bypass_layers=self.bypass_layers)


class BaseActivationBiasCorrectionPadNetTest(BaseActivationBiasCorrectionTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_quantization_config(self):
        # A small value set to the activation bias correction threshold only to activate the threshold
        # filtering without changing the bias correction values.
        return mct.core.QuantizationConfig(weights_bias_correction=False,
                                           weights_second_moment_correction=False,
                                           activation_bias_correction=True,
                                           activation_bias_correction_threshold=1e-6)

    def create_networks(self):
        return ActivationBiasCorrectionPadNet()


class BaseActivationBiasCorrectionBigThrTest(BaseActivationBiasCorrectionTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_quantization_config(self):
        # A large value is assigned to the activation bias correction threshold to enable threshold filtering,
        # which adjusts the bias correction values to zero.
        return mct.core.QuantizationConfig(weights_bias_correction=False,
                                           weights_second_moment_correction=False,
                                           activation_bias_correction=True,
                                           activation_bias_correction_threshold=1e9)

    def create_networks(self):
        return ActivationBiasCorrectionPadNet()

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        bias = float_model.linear_layer.bias.cpu().detach().numpy()
        bias_after_activation_bias_correction = quantized_model.linear_layer.layer.bias.cpu().detach().numpy()
        self.unit_test.assertTrue(np.array_equal(bias, bias_after_activation_bias_correction),
                                  msg=f"Error in activation bias correction: expected no change in the bias value in "
                                      f"case of activation_bias_correction_threshold 1e9.")


class BaseActivationBiasCorrectionReshapeNetTest(BaseActivationBiasCorrectionTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_networks(self):
        return ActivationBiasCorrectionReshapeNet()

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
import inspect

import unittest

import random
from torch.fx import symbolic_trace

from mct_quantizers import PytorchActivationQuantizationHolder, PytorchQuantizationWrapper
from mct_quantizers.common.constants import ACTIVATION_HOLDER_QUANTIZER
from model_compression_toolkit.target_platform_capabilities.target_platform import TargetPlatformCapabilities
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import generate_pytorch_tpc
from model_compression_toolkit.core.pytorch.utils import set_model, to_torch_tensor, \
    torch_tensor_to_numpy
import model_compression_toolkit as mct
import torch
import numpy as np
from tests.common_tests.base_feature_test import BaseFeatureNetworkTest
from tests.common_tests.helpers.generate_test_tp_model import generate_test_tp_model
from tests.pytorch_tests.model_tests.base_pytorch_test import BasePytorchTest
from tests.pytorch_tests.model_tests.feature_models.mixed_precision_activation_test import \
    MixedPrecisionActivationBaseTest
from model_compression_toolkit.core.common.network_editors.node_filters import NodeTypeFilter, NodeNameFilter


class NetForBitSelection(torch.nn.Module):
    def __init__(self, input_shape):
        super(NetForBitSelection, self).__init__()
        b, in_channels, h, w = input_shape[0]
        self.conv1 = torch.nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1))
        self.bn1 = torch.nn.BatchNorm2d(in_channels)
        self.conv2 = torch.nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1))
        self.relu = torch.nn.ReLU()
        self.fc = torch.nn.Linear(in_channels * h * w, 5)

    def forward(self, inp):
        out1 = self.conv1(inp)
        out1 = out1 + 3
        x = self.bn1(out1)
        x = self.conv2(x)
        x = self.relu(x)
        x = x + out1
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output

def get_layer_type_from_activation_quantizer(model, layer_name):
    layer_for_act_quant_name = layer_name.split('_' + ACTIVATION_HOLDER_QUANTIZER)[0]
    for name, layer in model.named_modules():
        if name == layer_for_act_quant_name:
            if isinstance(layer, PytorchQuantizationWrapper):
                return layer.layer
            else:
                return layer

class BaseManualBitWidthSelectionTest(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_feature_network(self, input_shape):
        return NetForBitSelection(input_shape)

    def get_mp_core_config(self):
        qc = mct.core.QuantizationConfig(mct.core.QuantizationErrorMethod.MSE, mct.core.QuantizationErrorMethod.MSE,
                                         relu_bound_to_power_of_2=False, weights_bias_correction=True,
                                         input_scaling=False, activation_channel_equalization=False)
        mpc = mct.core.MixedPrecisionQuantizationConfig(num_of_images=1)

        core_config = mct.core.CoreConfig(quantization_config=qc, mixed_precision_config=mpc)
        return core_config


class ManualBitWidthByLayerTypeTest(BaseManualBitWidthSelectionTest):
    def __init__(self, unit_test, filters, bit_widths):
        self.filters = filters
        if not isinstance(self.filters, list):
            self.filters = [self.filters]
        self.bit_widths = bit_widths
        if not isinstance(self.bit_widths, list):
            self.bit_widths = [self.bit_widths]
        # self.node_types = [filter.node_type for filter in self.filters]
        self.layer_types = {}
        self.functional_names = {}
        for filter, bit_width in zip(self.filters, self.bit_widths):
            if inspect.isclass(filter.node_type):
                self.layer_types.update({filter.node_type: bit_width})
            else:
                self.functional_names.update({filter.node_type.__name__: bit_width})

        super().__init__(unit_test)
    # def get_tpc(self):
    #     return {'all_8bit': generate_pytorch_tpc(name="8_quant_pytorch_test",
    #                                      tp_model=generate_test_tp_model({'weights_n_bits': 8,
    #                                                                       'activation_n_bits': 8,
    #                                                                       'enable_weights_quantization': True,
    #                                                                       'enable_activation_quantization': True
    #                                                                       })),
    #             }
    def get_core_configs(self):
        core_config = super().get_mp_core_config()
        for filter, bit_width in zip(self.filters, self.bit_widths):
            core_config.bit_width_config.set_manual_activation_bit_width(filter, bit_width)
        return {"mixed_precision_activation_model": core_config}

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        for model_name, quantized_model in quantized_models.items():
            for name, layer in quantized_model.named_modules():
                if isinstance(layer, PytorchActivationQuantizationHolder):
                    layer_type = get_layer_type_from_activation_quantizer(quantized_model, name)
                    if self.layer_types.get(type(layer_type)) is not None:
                        self.unit_test.assertTrue(layer.activation_holder_quantizer.num_bits == self.layer_types.get(type(layer_type)))
                    elif any([k in name for k in self.functional_names.keys()]):
                        for k in self.functional_names.keys():
                            if k in name:
                                bit_width = self.functional_names.get(k)
                        self.unit_test.assertTrue(layer.activation_holder_quantizer.num_bits == bit_width)
                    else:
                        self.unit_test.assertFalse(layer.activation_holder_quantizer.num_bits in self.bit_widths, msg=f"name {name}, layer.activation_holder_quantizer.num_bits {layer.activation_holder_quantizer.num_bits }, {self.bit_widths}")


class ManualBitWidthByFunctionalLayerTest(ManualBitWidthByLayerTypeTest):
    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        for model_name, quantized_model in quantized_models.items():
            for name, layer in quantized_model.named_modules():
                if isinstance(layer, PytorchActivationQuantizationHolder):
                    if any([n in name for n in self.functional_names]):
                        for index , n in enumerate(self.functional_names):
                            if n in name:
                                break
                        self.unit_test.assertTrue(layer.activation_holder_quantizer.num_bits == self.bit_widths[index])
                    else:
                        self.unit_test.assertFalse(layer.activation_holder_quantizer.num_bits in self.bit_widths)

# class InvalidBitWidthSelectionTest(BaseManualBitWidthSelectionTest):

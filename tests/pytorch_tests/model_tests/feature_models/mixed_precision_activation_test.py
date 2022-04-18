# Copyright 2022 Sony Semiconductors Israel, Inc. All rights reserved.
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
from torch.nn import Conv2d

from model_compression_toolkit import MixedPrecisionQuantizationConfig, KPI
from model_compression_toolkit.common.user_info import UserInformation
from model_compression_toolkit.hardware_models.default_hwm import get_default_hardware_model, \
    get_op_quantization_configs
from tests.common_tests.helpers.activation_mp_hw_model import generate_hw_model_with_activation_mp
from tests.common_tests.helpers.generate_test_hw_model import generate_test_hw_model, \
    generate_mixed_precision_test_hw_model
from tests.pytorch_tests.fw_hw_model_pytorch import get_mp_activation_pytorch_hwm_dict
from tests.pytorch_tests.layer_tests.base_pytorch_layer_test import get_layer_test_fw_hw_model_dict
from tests.pytorch_tests.model_tests.base_pytorch_test import BasePytorchTest
import model_compression_toolkit as mct

"""
This test checks the Mixed Precision feature.
"""


class MixedPercisionActivationBaseTest(BasePytorchTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_fw_hw_model(self):
        base_config, _ = get_op_quantization_configs()
        # return get_mp_activation_pytorch_hwm_dict
        return get_mp_activation_pytorch_hwm_dict(
            hardware_model=generate_hw_model_with_activation_mp(
                base_cfg=base_config,
                mp_bitwidth_candidates_list=[(8, 8), (8, 4), (8, 2),
                                             (4, 8), (4, 4), (4, 2),
                                             (2, 8), (2, 4), (2, 2)]),
            test_name='mixed_precision_activation_model',
            fhwm_name='mixed_precision_activation_pytorch_test')

    def get_quantization_configs(self):
        qc = mct.QuantizationConfig(mct.QuantizationErrorMethod.MSE,
                                    mct.QuantizationErrorMethod.MSE,
                                    weights_bias_correction=True,
                                    weights_per_channel_threshold=True,
                                    activation_channel_equalization=False,
                                    relu_bound_to_power_of_2=False,
                                    input_scaling=False)

        return {"mixed_precision_activation_model": MixedPrecisionQuantizationConfig(qc, num_of_images=1)}

    def create_feature_network(self, input_shape):
        return MixedPrecisionNet(input_shape)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info: UserInformation = None):
        # This is a base test, so it does not check a thing. Only actual tests of mixed precision
        # compare things to test.
        raise NotImplementedError

    def get_split_candidates(self, mp_config, weights_layers_idx, activation_layers_idx, model_layers):
        fw_hw_model = self.get_fw_hw_model()
        layers_to_quantize = [layer for layer in model_layers
                              if type(layer) not in fw_hw_model.get_layers_by_opset_name("NoQuantization")]
        layer2qco = fw_hw_model.layer2qco

        # get sorted candidates of each layer
        activation_layers_candidates = np.array(layers_to_quantize)[activation_layers_idx]
        weights_layers_candidates = np.array(layers_to_quantize)[weights_layers_idx]
        activation_candidates = [[(qc.weights_n_bits, qc.activation_n_bits) for qc in
                                  layer2qco.get(type(layer)).quantization_config_list] for layer in
                                 activation_layers_candidates]
        weights_candidates = [[(qc.weights_n_bits, qc.activation_n_bits) for qc in
                               layer2qco.get(type(layer)).quantization_config_list] for layer in
                              weights_layers_candidates]

        for layer_candidates in activation_candidates:
            layer_candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)

        for layer_candidates in weights_candidates:
            layer_candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)

        # get chosen n_bits for each layer (weights and activation separately)
        # NOTE: we assume that the order of the layers in the configuration is the same as it appears in model.layers,
        #   if this not the case, then this helper test function isn't valid.
        activation_bits = [activation_candidates[i][bitwidth_idx][1] for i, bitwidth_idx in
                           enumerate(np.array(mp_config)[activation_layers_idx])]
        weights_bits = [weights_candidates[i][bitwidth_idx][0] for i, bitwidth_idx in
                        enumerate(np.array(mp_config)[weights_layers_idx])]

        return weights_bits, activation_bits

    def get_split_candidates(self, quantization_info, quantized_models, activation_layers_idxs, weights_layers_idxs):
        quantized_model = quantized_models['mixed_precision_activation_model']
        fw_hw_model = self.get_fw_hw_model()['mixed_precision_activation_model']
        quantized_model_layers = list(quantized_model.children())
        layers_with_activations = [layer for layer in quantized_model_layers
                                   if type(layer) in fw_hw_model.get_layers_by_opset_name('Weights_n_Activation')
                                   or type(layer) in fw_hw_model.get_layers_by_opset_name('Activation')]
        layers_with_weights = [layer for layer in quantized_model_layers
                               if type(layer) in fw_hw_model.get_layers_by_opset_name('Weights_n_Activation')]

        layer2qco = fw_hw_model.layer2qco

        # Get layers' bitwidth candidates
        activation_candidates = [[(qc.weights_n_bits, qc.activation_n_bits) for qc in
                                  layer2qco.get(type(layer)).quantization_config_list] for layer in
                                 layers_with_activations]
        weights_candidates = [[(qc.weights_n_bits, qc.activation_n_bits) for qc in
                               layer2qco.get(type(layer)).quantization_config_list] for layer in
                              layers_with_weights]

        for layer_candidates in activation_candidates:
            layer_candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)

        for layer_candidates in weights_candidates:
            layer_candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)

        # get chosen n_bits for each layer (weights and activation separately)
        # NOTE: we assume that the order of the layers in the configuration is the same as it appears in model.layers,
        #   if this not the case, then this helper test function isn't valid.
        activation_bits = [activation_candidates[i][bitwidth_idx][1] for i, bitwidth_idx in
                           enumerate(np.array(quantization_info.mixed_precision_cfg)[activation_layers_idxs])]
        weights_bits = [weights_candidates[i][bitwidth_idx][0] for i, bitwidth_idx in
                        enumerate(np.array(quantization_info.mixed_precision_cfg)[weights_layers_idxs])]

        return weights_bits, activation_bits


class MixedPercisionActivationSearch8Bit(MixedPercisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_kpi(self):
        return KPI(np.inf, np.inf)

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        weights_bits, activation_bits = self.get_split_candidates(quantization_info, quantized_models,
                                                                  activation_layers_idxs=[0, 1, 3],
                                                                  weights_layers_idxs=[1, 2])
        self.unit_test.assertTrue((activation_bits == [8, 8, 8]))
        self.unit_test.assertTrue((weights_bits == [8, 8]))


class MixedPercisionActivationSearch2Bit(MixedPercisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_kpi(self):
        return KPI(96, 2119)

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        weights_bits, activation_bits = self.get_split_candidates(quantization_info, quantized_models,
                                                                  activation_layers_idxs=[0, 1, 3],
                                                                  weights_layers_idxs=[1, 2])
        self.unit_test.assertTrue((activation_bits == [2, 2, 2]))
        self.unit_test.assertTrue((weights_bits == [2, 2]))


class MixedPercisionActivationSearch4Bit(MixedPercisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_kpi(self):
        return KPI(192, 4238)

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        weights_bits, activation_bits = self.get_split_candidates(quantization_info, quantized_models,
                                                                  activation_layers_idxs=[0, 1, 3],
                                                                  weights_layers_idxs=[1, 2])
        self.unit_test.assertTrue(any(i <= 4 for i in activation_bits))
        self.unit_test.assertTrue(any(i <= 4 for i in weights_bits))


class MixedPrecisionNet(torch.nn.Module):
    def __init__(self, input_shape):
        super(MixedPrecisionNet, self).__init__()
        _, in_channels, _, _ = input_shape[0]
        self.conv1 = torch.nn.Conv2d(in_channels, 3, kernel_size=3)
        self.bn1 = torch.nn.BatchNorm2d(3)
        self.conv2 = torch.nn.Conv2d(3, 4, kernel_size=5)
        self.relu = torch.nn.ReLU()

    def forward(self, inp):
        x = self.conv1(inp)
        x = self.bn1(x)
        x = self.conv2(x)
        output = self.relu(x)
        return output

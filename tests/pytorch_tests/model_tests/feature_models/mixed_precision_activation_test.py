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
from torch import softmax, sigmoid
from torch.nn import Softmax, Sigmoid

from model_compression_toolkit import DefaultDict
from model_compression_toolkit.core import MixedPrecisionQuantizationConfig, ResourceUtilization
from model_compression_toolkit.core.common.user_info import UserInformation
from model_compression_toolkit.target_platform_capabilities.constants import KERNEL_ATTR, BIAS_ATTR, PYTORCH_KERNEL, \
    BIAS
from model_compression_toolkit.target_platform_capabilities.target_platform import QuantizationConfigOptions, \
    TargetPlatformModel, OperatorsSet, TargetPlatformCapabilities, OperationsSetToLayers
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import get_op_quantization_configs
from tests.common_tests.helpers.generate_test_tp_model import generate_tp_model_with_activation_mp
from tests.pytorch_tests.model_tests.base_pytorch_test import BasePytorchTest
import model_compression_toolkit as mct
from tests.pytorch_tests.tpc_pytorch import get_mp_activation_pytorch_tpc_dict

"""
This test checks the Mixed Precision feature.
"""


class MixedPrecisionActivationBaseTest(BasePytorchTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_tpc(self):
        base_config, _, default_config = get_op_quantization_configs()
        return get_mp_activation_pytorch_tpc_dict(
            tpc_model=generate_tp_model_with_activation_mp(
                base_cfg=base_config,
                default_config=default_config,
                mp_bitwidth_candidates_list=[(8, 8), (8, 4), (8, 2),
                                             (4, 8), (4, 4), (4, 2),
                                             (2, 8), (2, 4), (2, 2)]),
            test_name='mixed_precision_activation_model',
            tpc_name='mixed_precision_activation_pytorch_test')

    def get_core_configs(self):
        qc = mct.core.QuantizationConfig(mct.core.QuantizationErrorMethod.MSE, mct.core.QuantizationErrorMethod.MSE,
                                         relu_bound_to_power_of_2=False, weights_bias_correction=True,
                                         input_scaling=False, activation_channel_equalization=False)
        mpc = mct.core.MixedPrecisionQuantizationConfig(num_of_images=1)

        return {"mixed_precision_activation_model": mct.core.CoreConfig(quantization_config=qc, mixed_precision_config=mpc)}

    def create_feature_network(self, input_shape):
        return MixedPrecisionNet(input_shape)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info: UserInformation = None):
        # This is a base test, so it does not check a thing. Only actual tests of mixed precision
        # compare things to test.
        raise NotImplementedError

    def verify_config(self, result_config, expected_config):
        self.unit_test.assertTrue(all(result_config == expected_config))


class MixedPrecisionActivationSearch8Bit(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)
        self.expected_config = [1, 0, 0]

    def get_resource_utilization(self):
        return ResourceUtilization(np.inf, 3000)

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        self.verify_config(quantization_info.mixed_precision_cfg, self.expected_config)


class MixedPrecisionActivationSearch2Bit(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)
        self.expected_config = [2, 8, 2, 2]

    def get_resource_utilization(self):
        return ResourceUtilization(96, 768)

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        self.verify_config(quantization_info.mixed_precision_cfg, self.expected_config)


class MixedPrecisionActivationSearch4Bit(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)
        self.expected_config = [1, 4, 1, 1]

    def get_resource_utilization(self):
        return ResourceUtilization(192, 1536)

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        self.verify_config(quantization_info.mixed_precision_cfg, self.expected_config)


class MixedPrecisionActivationSearch4BitFunctional(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)
        self.expected_config = [1, 4, 4, 1]

    def get_resource_utilization(self):
        return ResourceUtilization(81, 1536)

    def create_feature_network(self, input_shape):
        return MixedPrecisionFunctionalNet(input_shape)

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        self.verify_config(quantization_info.mixed_precision_cfg, self.expected_config)


class MixedPrecisionActivationMultipleInputs(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)
        self.expected_config = [0 for _ in range(8)] + [1] # expected config for this test.
        self.num_calibration_iter = 3
        self.val_batch_size = 2

    def get_resource_utilization(self):
        return ResourceUtilization(np.inf, 431)

    def get_tpc(self):
        base_config, _, default_config = get_op_quantization_configs()
        return get_mp_activation_pytorch_tpc_dict(
            tpc_model=generate_tp_model_with_activation_mp(
                base_cfg=base_config,
                default_config=default_config,
                mp_bitwidth_candidates_list=[(8, 8), (8, 4), (8, 2),
                                             (4, 8), (4, 4), (4, 2),
                                             (2, 8), (2, 4), (2, 2)],
            custom_opsets=['Concat']),
            custom_opsets_to_layer={'Concat': [torch.concat]},
            test_name='mixed_precision_activation_model',
            tpc_name='mixed_precision_activation_pytorch_test')

    def get_mixed_precision_config(self):
        return MixedPrecisionQuantizationConfig(num_of_images=4)

    def create_feature_network(self, input_shape):
        return MixedPrecisionMultipleInputsNet(input_shape)

    def create_inputs_shape(self):
        return [[self.val_batch_size, 1, 8, 8],
                [self.val_batch_size, 1, 8, 8],
                [self.val_batch_size, 1, 8, 8],
                [self.val_batch_size, 1, 8, 8]]

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        self.verify_config(quantization_info.mixed_precision_cfg, self.expected_config)


class MixedPrecisionNet(torch.nn.Module):
    def __init__(self, input_shape):
        super(MixedPrecisionNet, self).__init__()
        _, in_channels, _, _ = input_shape[0]
        self.conv1 = torch.nn.Conv2d(in_channels, 3, kernel_size=(3, 3))
        self.bn1 = torch.nn.BatchNorm2d(3)
        self.conv2 = torch.nn.Conv2d(3, 4, kernel_size=(5, 5))
        self.relu = torch.nn.ReLU()

    def forward(self, inp):
        x = self.conv1(inp)
        x = self.bn1(x)
        x = self.conv2(x)
        output = self.relu(x)
        return output


class MixedPrecisionFunctionalNet(torch.nn.Module):
    def __init__(self, input_shape):
        super(MixedPrecisionFunctionalNet, self).__init__()
        _, in_channels, _, _ = input_shape[0]
        self.conv1 = torch.nn.Conv2d(in_channels, 3, kernel_size=(3, 3))
        self.conv2 = torch.nn.Conv2d(in_channels, 3, kernel_size=(3, 3))

    def forward(self, inp):
        x1 = self.conv1(inp)
        x2 = self.conv2(inp)
        output = x1 + x2
        return output


class MixedPrecisionMultipleInputsNet(torch.nn.Module):
    def __init__(self, input_shape):
        super(MixedPrecisionMultipleInputsNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 3, kernel_size=(3, 3))
        self.conv2 = torch.nn.Conv2d(1, 3, kernel_size=(3, 3))
        self.conv3 = torch.nn.Conv2d(1, 3, kernel_size=(3, 3))
        self.conv4 = torch.nn.Conv2d(1, 3, kernel_size=(3, 3))

    def forward(self, x, y, z, w):
        x1 = self.conv1(x)
        x2 = self.conv2(y)
        x3 = self.conv3(z)
        x4 = self.conv4(w)
        return torch.concat([x1, x2, x3, x4], dim=1)


class MixedPrecisionActivationTestNet(torch.nn.Module):
    def __init__(self, input_shape):
        super(MixedPrecisionActivationTestNet, self).__init__()
        _, in_channels, _, _ = input_shape[0]
        self.conv1 = torch.nn.Conv2d(in_channels, 3, kernel_size=(3, 3))
        self.relu = torch.nn.ReLU()

    def forward(self, inp):
        x = self.conv1(inp)
        x = torch.add(x, x)
        output = self.relu(x)
        return output


class MixedPrecisionDistanceFunctionsNet(torch.nn.Module):
    def __init__(self, input_shape):
        super(MixedPrecisionDistanceFunctionsNet, self).__init__()
        self.softmax = Softmax(dim=-1)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        x = self.softmax(x)
        x = self.sigmoid(x)
        x = softmax(x, dim=-1)
        x = sigmoid(x)

        return x


class MixedPrecisionDistanceFunctions(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)
        self.expected_config = [1, 1, 1, 1, 1]

    def get_resource_utilization(self):
        return ResourceUtilization(np.inf, 3071)

    def get_tpc(self):
        base_config, _, default_config = get_op_quantization_configs()
        custom_opsets_to_layer = {'Softmax': [softmax, Softmax]}
        mp_list = [(8, 8), (8, 4), (8, 2),
                   (4, 8), (4, 4), (4, 2),
                   (2, 8), (2, 4), (2, 2)]

        tp_model = generate_tp_model_with_activation_mp(
            base_cfg=base_config,
            default_config=default_config,
            mp_bitwidth_candidates_list=mp_list,
            custom_opsets=['Softmax'])

        return get_mp_activation_pytorch_tpc_dict(tpc_model=tp_model,
                                                  custom_opsets_to_layer=custom_opsets_to_layer,
                                                  test_name='mixed_precision_activation_model',
                                                  tpc_name='mixed_precision_distance_fn_test')

    def create_feature_network(self, input_shape):
        return MixedPrecisionDistanceFunctionsNet(input_shape)

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        self.verify_config(quantization_info.mixed_precision_cfg, self.expected_config)


class MixedPrecisionActivationConfigurableWeights(MixedPrecisionActivationBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)
        self.expected_config = [1, 1]

    def get_tpc(self):
        cfg, mixed_precision_cfg_list, _ = get_op_quantization_configs()

        act_eight_bit_cfg = cfg.clone_and_edit(activation_n_bits=8,
                                               attr_weights_configs_mapping={})
        act_four_bit_cfg = cfg.clone_and_edit(activation_n_bits=4,
                                              attr_weights_configs_mapping={})
        act_two_bit_cfg = cfg.clone_and_edit(activation_n_bits=2,
                                             attr_weights_configs_mapping={})

        mixed_precision_cfg_list = \
            [c.clone_and_edit(enable_activation_quantization=False) for c in mixed_precision_cfg_list]
        cfg = mixed_precision_cfg_list[0]

        act_mixed_cfg = QuantizationConfigOptions(
            [act_eight_bit_cfg, act_four_bit_cfg, act_two_bit_cfg],
            base_config=act_eight_bit_cfg,
        )

        weight_mixed_cfg = QuantizationConfigOptions(
            mixed_precision_cfg_list,
            base_config=cfg,
        )

        tp_model = TargetPlatformModel(QuantizationConfigOptions([cfg], cfg),
                                       name="mp_activation_conf_weights_test")

        with tp_model:
            OperatorsSet("Activations", act_mixed_cfg)
            OperatorsSet("Weights", weight_mixed_cfg)

        torch_tpc = TargetPlatformCapabilities(tp_model, name="mp_activation_conf_weights_test")

        with torch_tpc:
            OperationsSetToLayers(
                "Weights",
                [torch.nn.Conv2d],
                attr_mapping={KERNEL_ATTR: DefaultDict(default_value=PYTORCH_KERNEL),
                              BIAS_ATTR: DefaultDict(default_value=BIAS)}
            )

            OperationsSetToLayers(
                "Activations",
                [torch.nn.ReLU, torch.add]
            )

        return {'mixed_precision_activation_model': torch_tpc}

    def create_feature_network(self, input_shape):
        return MixedPrecisionActivationTestNet(input_shape)

    def get_resource_utilization(self):
        return ResourceUtilization(np.inf, 1536)

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        self.verify_config(quantization_info.mixed_precision_cfg, self.expected_config)

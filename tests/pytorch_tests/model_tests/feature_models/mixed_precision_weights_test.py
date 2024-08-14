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
from torch.nn import Conv2d

from model_compression_toolkit.defaultdict import DefaultDict
from model_compression_toolkit.core import ResourceUtilization
from model_compression_toolkit.core.common.mixed_precision.distance_weighting import MpDistanceWeighting
from model_compression_toolkit.core.common.user_info import UserInformation
from model_compression_toolkit.core.pytorch.constants import BIAS
from model_compression_toolkit.target_platform_capabilities.constants import KERNEL_ATTR, PYTORCH_KERNEL, BIAS_ATTR
from model_compression_toolkit.target_platform_capabilities.target_platform import QuantizationConfigOptions, \
    TargetPlatformModel, OperatorsSet, TargetPlatformCapabilities, OperationsSetToLayers
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import get_tp_model, get_op_quantization_configs
from tests.common_tests.helpers.generate_test_tp_model import generate_mixed_precision_test_tp_model
from tests.pytorch_tests.tpc_pytorch import get_pytorch_test_tpc_dict
from tests.pytorch_tests.model_tests.base_pytorch_test import BasePytorchTest
import model_compression_toolkit as mct

tp = mct.target_platform

"""
This test checks the Mixed Precision feature.
"""


class MixedPrecisionBaseTest(BasePytorchTest):
    def __init__(self, unit_test, num_calibration_iter=1):
        super().__init__(unit_test, num_calibration_iter=num_calibration_iter)

    def get_tpc(self):
        return get_pytorch_test_tpc_dict(tp_model=get_tp_model(),
                                         test_name='mixed_precision_model',
                                         ftp_name='mixed_precision_pytorch_test')

    def get_core_configs(self):
        qc = mct.core.QuantizationConfig(mct.core.QuantizationErrorMethod.MSE, mct.core.QuantizationErrorMethod.MSE,
                                         relu_bound_to_power_of_2=False, weights_bias_correction=True,
                                         input_scaling=False, activation_channel_equalization=False)
        mpc = mct.core.MixedPrecisionQuantizationConfig(num_of_images=1)

        return {"mixed_precision_model": mct.core.CoreConfig(quantization_config=qc, mixed_precision_config=mpc)}

    def create_feature_network(self, input_shape):
        return MixedPrecisionNet(input_shape)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info: UserInformation = None):
        # This is a base test, so it does not check a thing. Only actual tests of mixed precision
        # compare things to test.
        raise NotImplementedError

    def compare_results(self, quantization_info, quantized_models, float_model, expected_bitwidth_idx):
        # verify that quantization occurred
        quantized_model = quantized_models['mixed_precision_model']
        conv_layers = list(filter(lambda _layer: type(_layer) == Conv2d, list(quantized_model.children())))
        float_conv_layers = list(filter(lambda _layer: type(_layer) == Conv2d, list(float_model.children())))
        for idx, layer in enumerate(conv_layers):  # quantized per channel
            q_weights = layer.weight.detach().cpu().numpy()
            float_weights = float_conv_layers[idx].weight.detach().cpu().numpy()
            for i in range(3):
                self.unit_test.assertTrue(
                    np.unique(q_weights[i, :, :, :]).flatten().shape[0] <= 2 ** [8, 4, 2][expected_bitwidth_idx])
            # quantized_model and float_model are not equal
            self.unit_test.assertFalse((q_weights == float_weights).all())


class MixedPrecisionSearch8Bit(MixedPrecisionBaseTest):
    def __init__(self, unit_test, distance_metric=MpDistanceWeighting.AVG):
        super().__init__(unit_test)

        self.distance_metric = distance_metric

    def get_resource_utilization(self):
        return ResourceUtilization(380)

    def get_core_configs(self):
        qc = mct.core.QuantizationConfig(mct.core.QuantizationErrorMethod.MSE, mct.core.QuantizationErrorMethod.MSE,
                                         relu_bound_to_power_of_2=False, weights_bias_correction=True,
                                         input_scaling=False, activation_channel_equalization=False)
        mpc = mct.core.MixedPrecisionQuantizationConfig(num_of_images=1,
                                                        distance_weighting_method=self.distance_metric)

        return {"mixed_precision_model": mct.core.CoreConfig(quantization_config=qc, mixed_precision_config=mpc)}

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        self.compare_results(quantization_info, quantized_models, float_model, 0)


class MixedPrecisionWithHessianScores(MixedPrecisionBaseTest):
    def __init__(self, unit_test, distance_metric=MpDistanceWeighting.AVG):
        super().__init__(unit_test, num_calibration_iter=10)

        self.distance_metric = distance_metric

    def generate_inputs(self, input_shapes):
        return [np.random.random(in_shape) for in_shape in input_shapes]

    def get_resource_utilization(self):
        return ResourceUtilization(380)

    def get_core_configs(self):
        qc = mct.core.QuantizationConfig(mct.core.QuantizationErrorMethod.MSE, mct.core.QuantizationErrorMethod.MSE,
                                         relu_bound_to_power_of_2=False, weights_bias_correction=True,
                                         input_scaling=False, activation_channel_equalization=False)
        mpc = mct.core.MixedPrecisionQuantizationConfig(num_of_images=10,
                                                        distance_weighting_method=self.distance_metric,
                                                        use_hessian_based_scores=True)

        return {"mixed_precision_model": mct.core.CoreConfig(quantization_config=qc, mixed_precision_config=mpc)}

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        self.compare_results(quantization_info, quantized_models, float_model, 0)


class MixedPrecisionSearchPartWeightsLayers(MixedPrecisionBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_tpc(self):
        # Building a TPC that gives Conv layers mixed precision candidates and Dense layers a fixed candidate.
        # Both layers that have weights to quantized, so we want to verify that finalizing the model is successful.
        # Note that this is important that the quantization config options would include also activation quantization.
        cfg, mixed_precision_cfg_list, _ = get_op_quantization_configs()

        two_bit_cfg = mixed_precision_cfg_list[2]

        weight_mixed_cfg = tp.QuantizationConfigOptions(
            mixed_precision_cfg_list,
            base_config=cfg,
        )

        weight_fixed_cfg = tp.QuantizationConfigOptions(
            [two_bit_cfg],
            base_config=two_bit_cfg,
        )

        tp_model = tp.TargetPlatformModel(weight_fixed_cfg, name="mp_part_weights_layers_test")
        with tp_model:
            tp.OperatorsSet("Weights_mp", weight_mixed_cfg)
            tp.OperatorsSet("Weights_fixed", weight_fixed_cfg)

        pytorch_tpc = tp.TargetPlatformCapabilities(tp_model, name="mp_part_weights_layers_test")

        with pytorch_tpc:
            tp.OperationsSetToLayers(
                "Weights_fixed",
                [torch.nn.Linear],
                attr_mapping={KERNEL_ATTR: DefaultDict(default_value=PYTORCH_KERNEL),
                              BIAS_ATTR: DefaultDict(default_value=BIAS)}
            )

            tp.OperationsSetToLayers(
                "Weights_mp",
                [torch.nn.Conv2d],
                attr_mapping={KERNEL_ATTR: DefaultDict(default_value=PYTORCH_KERNEL),
                              BIAS_ATTR: DefaultDict(default_value=BIAS)}
            )

        return {'mixed_precision_model': pytorch_tpc}

    def create_feature_network(self, input_shape):
        class ConvLinearModel(torch.nn.Module):
            def __init__(self, _input_shape):
                super(ConvLinearModel, self).__init__()
                _, in_channels, _, _ = _input_shape[0]
                self.conv = torch.nn.Conv2d(in_channels, 3, kernel_size=3)
                self.linear = torch.nn.Linear(30, 64)

            def forward(self, inp):
                x = self.conv(inp)
                output = self.linear(x)
                return output

        return ConvLinearModel(input_shape)

    def get_resource_utilization(self):
        return ResourceUtilization(560)

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        # We just needed to verify that the graph finalization is working without failing.
        # The actual quantization is not interesting for the sake of this test, so we just verify some
        # degenerated things to see that everything worked.
        self.unit_test.assertTrue(
            quantization_info.mixed_precision_cfg == [1])  # resource utilization should give mixed precision

        quantized_model = quantized_models['mixed_precision_model']
        linear_layer = quantized_model.linear
        q_weights = linear_layer.get_quantized_weights()['weight'].detach().cpu().numpy()
        for i in range(q_weights.shape[0]):
            self.unit_test.assertTrue(
                np.unique(q_weights[i, :]).flatten().shape[0] <= 4)

class MixedPrecisionSearch2Bit(MixedPrecisionBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_resource_utilization(self):
        return ResourceUtilization(96)

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        self.compare_results(quantization_info, quantized_models, float_model, 2)


class MixedPrecisionSearch4Bit(MixedPrecisionBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_resource_utilization(self):
        return ResourceUtilization(192)

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        self.compare_results(quantization_info, quantized_models, float_model, 1)


class MixedPrecisionActivationDisabledTest(MixedPrecisionBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_fw_hw_model(self):
        base_config, _, default_config = get_op_quantization_configs()
        return get_pytorch_test_tpc_dict(
            tp_model=generate_mixed_precision_test_tp_model(
                base_cfg=base_config.clone_and_edit(enable_activation_quantization=False),
                default_config=default_config,
                mp_bitwidth_candidates_list=[(8, 8), (4, 8), (2, 8)]),
            test_name='mixed_precision_model',
            ftp_name='mixed_precision_pytorch_test')

    def get_resource_utilization(self):
        return ResourceUtilization(np.inf)

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        self.compare_results(quantization_info, quantized_models, float_model, 0)


class MixedPrecisionSearchLastLayerDistance(MixedPrecisionBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_resource_utilization(self):
        return ResourceUtilization(192)

    def get_mixed_precision_config(self):
        return mct.core.MixedPrecisionQuantizationConfig(num_of_images=1,
                                                         use_hessian_based_scores=False,
                                                         distance_weighting_method=MpDistanceWeighting.LAST_LAYER)

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        self.compare_results(quantization_info, quantized_models, float_model, 1)


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


class MixedPrecisionWeightsConfigurableActivations(MixedPrecisionBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)
        self.expected_config = [1]

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
                                       name="mp_weights_conf_act_test")

        with tp_model:
            OperatorsSet("Activations", act_mixed_cfg)
            OperatorsSet("Weights", weight_mixed_cfg)

        torch_tpc = TargetPlatformCapabilities(tp_model, name="mp_weights_conf_act_test")

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

        return {'mixed_precision_model': torch_tpc}

    def create_feature_network(self, input_shape):
        return MixedPrecisionWeightsTestNet(input_shape)

    def get_resource_utilization(self):
        return ResourceUtilization(80)

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        self.unit_test.assertTrue(all(quantization_info.mixed_precision_cfg == self.expected_config))


class MixedPrecisionWeightsTestNet(torch.nn.Module):
    def __init__(self, input_shape):
        super(MixedPrecisionWeightsTestNet, self).__init__()
        _, in_channels, _, _ = input_shape[0]
        self.conv1 = torch.nn.Conv2d(in_channels, 3, kernel_size=(3, 3))
        self.relu = torch.nn.ReLU()

    def forward(self, inp):
        x = self.conv1(inp)
        x = torch.add(x, x)
        output = self.relu(x)
        return output
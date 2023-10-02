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

from model_compression_toolkit.core import KPI
from model_compression_toolkit.core.common.mixed_precision.distance_weighting import get_last_layer_weights
from model_compression_toolkit.core.common.user_info import UserInformation
from model_compression_toolkit.target_platform_capabilities.target_platform.quantization_format import \
    QuantizationFormat
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import get_tp_model, get_op_quantization_configs
from tests.common_tests.helpers.generate_test_tp_model import generate_mixed_precision_test_tp_model
from tests.pytorch_tests.tpc_pytorch import get_pytorch_test_tpc_dict
from tests.pytorch_tests.model_tests.base_pytorch_test import BasePytorchTest
import model_compression_toolkit as mct

tp = mct.target_platform

"""
This test checks the Mixed Precision feature.
"""


class MixedPercisionBaseTest(BasePytorchTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_tpc(self):
        return get_pytorch_test_tpc_dict(tp_model=get_tp_model(),
                                         test_name='mixed_precision_model',
                                         ftp_name='mixed_precision_pytorch_test')

    def get_core_configs(self):
        qc = mct.core.QuantizationConfig(mct.core.QuantizationErrorMethod.MSE,
                                    mct.core.QuantizationErrorMethod.MSE,
                                    weights_bias_correction=True,
                                    weights_per_channel_threshold=True,
                                    activation_channel_equalization=False,
                                    relu_bound_to_power_of_2=False,
                                    input_scaling=False)
        mpc = mct.core.MixedPrecisionQuantizationConfigV2(num_of_images=1)

        return {"mixed_precision_model": mct.core.CoreConfig(quantization_config=qc, mixed_precision_config=mpc)}

    def create_feature_network(self, input_shape):
        return MixedPrecisionNet(input_shape)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info: UserInformation = None):
        # This is a base test, so it does not check a thing. Only actual tests of mixed precision
        # compare things to test.
        raise NotImplementedError

    def compare_results(self, quantization_info, quantized_models, float_model, expected_bitwidth_idx):
        # quantized with the highest precision since KPI==inf
        self.unit_test.assertTrue((quantization_info.mixed_precision_cfg ==
                                   [expected_bitwidth_idx, expected_bitwidth_idx]).all())
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


class MixedPercisionSearch8Bit(MixedPercisionBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_kpi(self):
        return KPI(np.inf)

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        self.compare_results(quantization_info, quantized_models, float_model, 0)


class MixedPercisionSearchPartWeightsLayers(MixedPercisionBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_tpc(self):
        # Building a TPC that gives Conv layers mixed precision candidates and Dense layers a fixed candidate.
        # Both layers that have weights to quantized, so we want to verify that finalizing the model is successful.
        # Note that this is important that the quantization config options would include also activation quantization.
        cfg, mixed_precision_cfg_list = get_op_quantization_configs()

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
            tp_model.set_quantization_format(QuantizationFormat.FAKELY_QUANT)

            tp.OperatorsSet("Weights_mp", weight_mixed_cfg)
            tp.OperatorsSet("Weights_fixed", weight_fixed_cfg)

        pytorch_tpc = tp.TargetPlatformCapabilities(tp_model, name="mp_part_weights_layers_test")

        with pytorch_tpc:
            tp.OperationsSetToLayers(
                "Weights_fixed",
                [torch.nn.Linear],
            )

            tp.OperationsSetToLayers(
                "Weights_mp",
                [torch.nn.Conv2d],
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

    def get_kpi(self):
        return KPI(np.inf)

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        # We just needed to verify that the graph finalization is working without failing.
        # The actual quantization is not interesting for the sake of this test, so we just verify some
        # degenerated things to see that everything worked.
        self.unit_test.assertTrue(
            quantization_info.mixed_precision_cfg == [0])  # kpi is infinity -> should give best model - 8bits

        quantized_model = quantized_models['mixed_precision_model']
        linear_layer = quantized_model.linear
        q_weights = linear_layer.get_quantized_weights()['weight'].detach().cpu().numpy()
        for i in range(q_weights.shape[0]):
            self.unit_test.assertTrue(
                np.unique(q_weights[i, :]).flatten().shape[0] <= 4)

class MixedPercisionSearch2Bit(MixedPercisionBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_kpi(self):
        return KPI(96)

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        self.compare_results(quantization_info, quantized_models, float_model, 2)


class MixedPercisionSearch4Bit(MixedPercisionBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_kpi(self):
        return KPI(192)

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        self.compare_results(quantization_info, quantized_models, float_model, 1)


class MixedPercisionActivationDisabledTest(MixedPercisionBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_fw_hw_model(self):
        base_config, _ = get_op_quantization_configs()
        return get_pytorch_test_tpc_dict(
            tp_model=generate_mixed_precision_test_tp_model(
                base_cfg=base_config.clone_and_edit(enable_activation_quantization=False),
                mp_bitwidth_candidates_list=[(8, 8), (4, 8), (2, 8)]),
            test_name='mixed_precision_model',
            ftp_name='mixed_precision_pytorch_test')

    def get_kpi(self):
        return KPI(np.inf)

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        self.compare_results(quantization_info, quantized_models, float_model, 0)


class MixedPercisionSearchLastLayerDistance(MixedPercisionBaseTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_kpi(self):
        return KPI(192)

    def get_mixed_precision_v2_config(self):
        return mct.core.MixedPrecisionQuantizationConfigV2(num_of_images=1,
                                                      use_grad_based_weights=False,
                                                      distance_weighting_method=get_last_layer_weights)

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

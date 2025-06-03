# Copyright 2025 Sony Semiconductor Israel, Inc. All rights reserved.
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
from typing import Callable, Union

import pytest
import torch

from model_compression_toolkit.core import ResourceUtilization, MpDistanceWeighting, MixedPrecisionQuantizationConfig, \
    CoreConfig
from model_compression_toolkit.ptq import pytorch_post_training_quantization
from model_compression_toolkit.target_platform_capabilities import QuantizationMethod, AttributeQuantizationConfig, \
    OpQuantizationConfig, QuantizationConfigOptions, Signedness, OperatorSetNames, \
    TargetPlatformCapabilities
from model_compression_toolkit.target_platform_capabilities.constants import KERNEL_ATTR, BIAS_ATTR
from tests_pytest._test_util.tpc_util import configure_mp_opsets_for_kernel_bias_ops, configure_mp_activation_opsets
from tests_pytest.pytorch_tests.torch_test_util.torch_test_mixin import BaseTorchIntegrationTest


class Model(torch.nn.Module):
    def __init__(self, input_shape):
        c, h, w = input_shape[-3:]
        super().__init__()
        self.conv1 = torch.nn.Conv2d(c, 8, kernel_size=3)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=5)
        self.relu2 = torch.nn.ReLU()
        self.fc = torch.nn.Linear(w-6, 20)

    def forward(self, inp):
        x = self.conv1(inp)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        y = self.fc(x)
        return y


@pytest.fixture
def input_shape():
    return 1, 3, 16, 16


@pytest.fixture
def model(input_shape):
    return Model(input_shape)


def build_tpc(default_a_bit: int, conv_a_bits: list, conv_w_bits: list, fc_a_bits: list, fc_w_bits: list, bn_a_bits: list):
    default_w_cfg = AttributeQuantizationConfig(weights_quantization_method=QuantizationMethod.POWER_OF_TWO,
                                                weights_n_bits=8,
                                                weights_per_channel_threshold=True,
                                                enable_weights_quantization=True)
    default_op_cfg = OpQuantizationConfig(
        default_weight_attr_config=default_w_cfg.clone_and_edit(enable_weights_quantization=False),
        attr_weights_configs_mapping={},
        activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
        activation_n_bits=default_a_bit,
        supported_input_activation_n_bits=[16, 8, 4, 2],
        enable_activation_quantization=True,
        quantization_preserving=False,
        fixed_scale=None, fixed_zero_point=None, simd_size=32, signedness=Signedness.AUTO)

    default_w_op_cfg = default_op_cfg.clone_and_edit(
        attr_weights_configs_mapping={KERNEL_ATTR: default_w_cfg, BIAS_ATTR: AttributeQuantizationConfig()}
    )
    default_cfg = QuantizationConfigOptions(quantization_configurations=[default_op_cfg])

    ops1, _ = configure_mp_opsets_for_kernel_bias_ops(opset_names=[OperatorSetNames.CONV],
                                                      base_w_config=default_w_cfg, base_op_config=default_w_op_cfg,
                                                      w_nbits=conv_w_bits, a_nbits=conv_a_bits)
    ops2, _ = configure_mp_opsets_for_kernel_bias_ops(opset_names=[OperatorSetNames.FULLY_CONNECTED],
                                                      base_w_config=default_w_cfg, base_op_config=default_w_op_cfg,
                                                      w_nbits=fc_w_bits, a_nbits=fc_a_bits)
    ops3, _ = configure_mp_activation_opsets(opset_names=[OperatorSetNames.BATCH_NORM], base_op_config=default_op_cfg,
                                             a_nbits=bn_a_bits)

    tpc = TargetPlatformCapabilities(default_qco=default_cfg, tpc_platform_type='test',
                                     operator_set=ops1 + ops2 + ops3, fusing_patterns=None)
    return tpc


def le(thresh):
    return lambda nb: nb <= thresh


class TestMixedPrecisionPTQ(BaseTorchIntegrationTest):
    """ Different ru targets are tested with configurable weights / activations / both.
        Bops is sanity-run with all distance metrics.
        No other configurations are tested here. """
    shape = (1, 3, 16, 16)

    sp_a_layers = ['inp', 'relu1', 'relu2']
    w_layers = ['conv1', 'conv2', 'fc']
    a_layers = sp_a_layers + w_layers + ['bn1']

    @pytest.fixture
    def datagen(self, input_shape):
        return self.get_basic_data_gen([input_shape])

    @pytest.mark.parametrize('w_ru, exp_res, eq_ru', [
        (216*2/8+3200*2/8+200*4/8, {'conv1': 2, 'conv2': 2, 'fc': 4}, True),  # min
        (216*4/8+3200*2/8+200*4/8-1, {'conv1': 2, 'conv2': 2, 'fc': 4}, False),  # largest min
        (216*4/8+3200*2/8+200*4/8, {'conv1': 4, 'conv2': 2, 'fc': 4}, True),  # second min
        (3416*16/8+200*8/8, {'conv1': 16, 'conv2': 16, 'fc': 8}, True),  # max
    ])
    def test_weights_mp(self, model, datagen, w_ru, exp_res, eq_ru):
        """ Test weights mixed-precision.
            All activations should use the default nbits, even when there is a higher candidate. """
        tpc = build_tpc(default_a_bit=4, conv_a_bits=[2, 4, 8, 16], conv_w_bits=[16, 8, 4, 2],
                        fc_a_bits=[2, 4, 8, 16], fc_w_bits=[4, 8], bn_a_bits=[2, 4, 8, 16])

        ru = ResourceUtilization(weights_memory=w_ru)
        qmodel, user_info = self._run(model, datagen, tpc, ru, 4, eq_ru)
        for layer in self.a_layers:
            self._validate_activation_nbits(qmodel, layer, 4)
        for layer_name, exp_nbit in exp_res.items():
            self._validate_weight_nbits(qmodel, layer_name, exp_nbit)

    @pytest.mark.parametrize('bops, exp_res, eq_ru', [
        (196*216*8*4 + 100*3200*8*4 + 160*200*8*2, {'conv1': 4, 'conv2': 4, 'fc': 2}, True),  # min
        (196*216*8*4 + 100*3200*8*4 + 160*200*8*4 - 1, {'conv1': 4, 'conv2': 4, 'fc': 2}, False),   # largest min
        (196*216*8*4 + 100*3200*8*4 + 160*200*8*4, {'conv1': 4, 'conv2': 4, 'fc': 4}, True),   # second min
        (196*216*8*16 + 100*3200*8*16 + 160*200*8*8, {'conv1': 16, 'conv2': 16, 'fc': 8}, True),   # max
    ])
    def test_bops_configurable_weights(self, model, datagen, bops, exp_res, eq_ru):
        """ Test bops target with configurable weights only in relevant nodes.
            Only activations that do not contribute to bops are configurable, and activation mp should be triggered by
            bops target => max nbit should be selected for them. """
        tpc = build_tpc(default_a_bit=8, conv_a_bits=[16, 8, 4, 2], conv_w_bits=[16, 8, 4],
                        fc_a_bits=[2, 4, 8, 16], fc_w_bits=[2, 4, 8], bn_a_bits=[8])
        ru = ResourceUtilization(bops=bops)
        qmodel, user_info = self._run(model, datagen, tpc, ru, 8, eq_ru)
        for layer in ['fc', 'conv1', 'conv2']:
            self._validate_activation_nbits(qmodel, layer, 16)
        for layer_name, exp_nbit in exp_res.items():
            self._validate_weight_nbits(qmodel, layer_name, exp_nbit)

    @pytest.mark.parametrize('bops, exp_res, eq_ru', [
        (196*216*4*8 + 100*3200*2*8 + 160*200*4*8, {'bn1': 2}, True),  # min
        # TODO: -1  (to -5) should all give the same result but pulp seems to return a wrong solution
        #  (https://github.com/coin-or/pulp/issues/822)
        (196*216*4*8 + 100*3200*4*8 + 160*200*4*8 - 6, {'bn1': 2}, False),  # second min
        (196*216*4*8 + 100*3200*4*8 + 160*200*4*8, {'bn1': 4}, True),  # second min
        (196*216*4*8 + 100*3200*16*8 + 160*200*4*8 - 1, {'bn1': le(8)}, False),  # max
        (196*216*4*8 + 100*3200*16*8 + 160*200*4*8, {'bn1': 16}, True),  # max
    ])
    def test_bops_configurable_activations(self, model, datagen, bops, exp_res, eq_ru):
        """ Test bops target with configurable activations.
            Only nodes contributing to bops should be affected, other activations should be set to max cfg."""
        tpc = build_tpc(default_a_bit=4, conv_a_bits=[4, 8, 16], conv_w_bits=[8],
                        fc_a_bits=[2, 4, 8, 16], fc_w_bits=[8], bn_a_bits=[16, 8, 4, 2])
        ru = ResourceUtilization(bops=bops)
        qmodel, user_info = self._run(model, datagen, tpc, ru, 4, eq_ru)
        for layer in self.w_layers:
            self._validate_weight_nbits(qmodel, layer, 8)
        # max cfg for act not affecting bops
        for layer in ['conv1', 'conv2', 'fc']:
            self._validate_activation_nbits(qmodel, layer, 16)
        for layer_name, exp_nbit in exp_res.items():
            self._validate_activation_nbits(qmodel, layer_name, exp_nbit)

    @pytest.mark.parametrize('bops, exp_w_res, exp_a_res, eq_ru', [
        (196*216*4*2 + 100*3200*2*2 + 160*200*4*4, {'conv1': 2, 'conv2': 2, 'fc': 4}, {'bn1': 2}, True),  # min
        (196*216*4*4 + 100*3200*2*2 + 160*200*4*4 - 1, {'conv1': 2, 'conv2': 2, 'fc': 4}, {'bn1': 2}, False),  # largest min
        (196*216*4*4 + 100*3200*2*2 + 160*200*4*4, {'conv1': 4, 'conv2': 2, 'fc': 4}, {'bn1': 2}, False),  # second min
        (196*216*4*8 + 100*3200*16*8 + 160*200*4*16, {'conv1': 8, 'conv2': 8, 'fc': 16}, {'bn1': 16}, True),  # max
    ])
    def test_bops_configurable_w_and_a(self, model, datagen, bops, exp_w_res, exp_a_res, eq_ru):
        """ Test bops with configurable weights and activations. """
        tpc = build_tpc(default_a_bit=4, conv_a_bits=[2, 4, 8, 16], conv_w_bits=[8, 4, 2],
                        fc_a_bits=[2, 4, 8, 16], fc_w_bits=[16, 8, 4], bn_a_bits=[16, 8, 4, 2])
        ru = ResourceUtilization(bops=bops)
        qmodel, user_info = self._run(model, datagen, tpc, ru, 4, eq_ru)
        # max nbit for activations that don't affect bops
        for layer in ['conv1', 'conv2', 'fc']:
            self._validate_activation_nbits(qmodel, layer, 16)
        for layer_name, exp_nbit in exp_w_res.items():
            self._validate_weight_nbits(qmodel, layer_name, exp_nbit)
        for layer_name, exp_nbit in exp_a_res.items():
            self._validate_activation_nbits(qmodel, layer_name, exp_nbit)

    @pytest.mark.parametrize('dist', set(MpDistanceWeighting) - {MpDistanceWeighting.HESSIAN})
    def test_bops_with_distance_metric(self, model, datagen, dist):
        """ Sanity test for bops with all distance metrics. """
        mp_cfg = MixedPrecisionQuantizationConfig(distance_weighting_method=dist)
        core_cfg = CoreConfig(mixed_precision_config=mp_cfg)
        tpc = build_tpc(default_a_bit=4, conv_a_bits=[2, 4, 8, 16], conv_w_bits=[8, 4, 2],
                        fc_a_bits=[2, 4, 8, 16], fc_w_bits=[16, 8, 4], bn_a_bits=[16, 8, 4, 2])
        ru = ResourceUtilization(bops=196*216*4*4 + 100*3200*2*2 + 160*200*4*4)

        qmodel, user_info = self._run(model, datagen, tpc, ru, 4, True, core_cfg=core_cfg)
        for layer_name, exp_nbit in {'conv1': 4, 'conv2': 2, 'fc': 4}.items():
            self._validate_weight_nbits(qmodel, layer_name, exp_nbit)
        for layer_name, exp_nbit in {'bn1': 2}.items():
            self._validate_activation_nbits(qmodel, layer_name, exp_nbit)

    @pytest.mark.parametrize('a_ru, exp_res, eq_ru', [
        # min cfg => max cut relu2-fc, other activations cannot exceed the cut
        (1600*4/8 + 3200*2/8, {'conv1': le(4), 'bn1': le(4), 'conv2': le(4), 'fc': 2}, True),
        (1568*8/8 + 1600*16/8, {'conv1': 16, 'bn1': 8, 'conv2': 16, 'fc': 8}, True),  # max
    ])
    def test_activation_mp(self, model, datagen, a_ru, exp_res, eq_ru):
        """ Test activation mixed-precision. Weights MP should be off, so all weights should use the default nbit
            despite having a larger candidate. """
        # in    c1     r1    bn       c2    r2      fc
        # 4*768 *1568 4*1568  *1568  *1600  4*1600  *3200
        ru = ResourceUtilization(activation_memory=a_ru)
        tpc = build_tpc(default_a_bit=4, conv_a_bits=[2, 4, 8, 16], conv_w_bits=[16, 8, 4, 2],
                        fc_a_bits=[2, 4, 8, 16], fc_w_bits=[2, 4, 8], bn_a_bits=[8, 4, 2])
        qmodel, user_info = self._run(model, datagen, tpc, ru, 4, eq_ru)
        # should use default weight config since weight mp is off
        for layer in self.w_layers:
            self._validate_weight_nbits(qmodel, layer, 8)
        for layer, exp_bit in exp_res.items():
            self._validate_activation_nbits(qmodel, layer, exp_bit)

    def test_activation_just_below_max(self, model, datagen):
        """ Test that only one candidate is reduced when target ru is just below the max config ru. """
        ru = ResourceUtilization(activation_memory=1568*8/8 + 1600*16/8 - 1)
        tpc = build_tpc(default_a_bit=4, conv_a_bits=[2, 4, 8, 16], conv_w_bits=[16, 8, 4, 2],
                        fc_a_bits=[2, 4, 8, 16], fc_w_bits=[2, 4, 8], bn_a_bits=[8, 4, 2])
        qmodel, user_info = self._run(model, datagen, tpc, ru, 4, False)
        bn_bit = self.fetch_activation_holder_quantizer(qmodel, 'bn1').num_bits
        conv2_bit = self.fetch_activation_holder_quantizer(qmodel, 'conv2').num_bits
        # exactly one of the layers constituting max cut is reduced
        assert (bn_bit < 8 or conv2_bit < 16) and (bn_bit == 8 or conv2_bit == 16)
        self._validate_activation_nbits(qmodel, 'conv1', 16)
        self._validate_activation_nbits(qmodel, 'fc', 8)

    @pytest.mark.parametrize('ru, exp_a_res, eq_ru', [
        # same as activation ru test + const weight
        (1600 * 4 / 8 + 3200 * 2 / 8 + 3616*8/8, {'conv1': le(4), 'bn1': le(4), 'conv2': le(4), 'fc': 2}, True),
        (1568 * 8 / 8 + 1600 * 16 / 8 + 3616*8/8, {'conv1': 16, 'bn1': 8, 'conv2': 16, 'fc': 8}, True),  # max
    ])
    def test_total_mp_configurable_activation(self, model, datagen, ru, exp_a_res, eq_ru):
        """ Test total target with configurable activations. """
        ru = ResourceUtilization(total_memory=ru)
        tpc = build_tpc(default_a_bit=4, conv_a_bits=[2, 4, 8, 16], conv_w_bits=[8],
                        fc_a_bits=[2, 4, 8, 16], fc_w_bits=[8], bn_a_bits=[8, 4, 2])
        qmodel, user_info = self._run(model, datagen, tpc, ru, 4, eq_ru)
        for layer in self.w_layers:
            self._validate_weight_nbits(qmodel, layer, 8)
        for layer, exp_bit in exp_a_res.items():
            self._validate_activation_nbits(qmodel, layer, exp_bit)

    @pytest.mark.parametrize('ru, exp_w_res, eq_ru', [
        (216*4/8 + 3200*4/8 + 200*4/8 + 19200/8 - 1, {'conv1': 4, 'conv2': 4, 'fc': 2}, False),  # largest min
        (3416*16/8 + 200*8/8 + 19200/8, {'conv1': 16, 'conv2': 16, 'fc': 8}, True),  # max
    ])
    def test_total_mp_configurable_weight(self, model, datagen, ru, exp_w_res, eq_ru):
        """ Test total target with configurable weights. """
        ru = ResourceUtilization(total_memory=ru)
        tpc = build_tpc(default_a_bit=4, conv_a_bits=[4], conv_w_bits=[16, 8, 4],
                        fc_a_bits=[4], fc_w_bits=[2, 4, 8], bn_a_bits=[4])
        qmodel, user_info = self._run(model, datagen, tpc, ru, 4, eq_ru)
        for layer in self.a_layers:
            self._validate_activation_nbits(qmodel, layer, 4)
        for layer, exp_bit in exp_w_res.items():
            self._validate_weight_nbits(qmodel, layer, exp_bit)

    @pytest.mark.parametrize('ru, exp_a_res, exp_w_res, eq_ru', [
        (1600*4/8 + 3200*2/8 + 3416*2/8 + 200*2/8, {'conv1': le(4), 'bn1': le(4), 'conv2': le(4), 'fc': 2},
         {'conv1': 2, 'conv2': 2, 'fc': 2}, True),
        (1600*4/8 + 3200*2/8 + 3416*2/8 + 200*4/8, {'conv1': le(4), 'bn1': le(4), 'conv2': le(4), 'fc': 2},
         {'conv1': 2, 'conv2': 2, 'fc': 4}, False),
        (1568*8/8 + 1600*16/8 + 3416*16/8 + 200*8/8, {'conv1': 16, 'bn1': 8, 'conv2': 16, 'fc': 8},
         {'conv1': 16, 'conv2': 16, 'fc': 8}, True),  # max
    ])
    def test_total_mp_configurable_w_and_a(self, model, datagen, ru, exp_a_res, exp_w_res, eq_ru):
        """ Test total target with both weight and activation being configurable. """
        ru = ResourceUtilization(total_memory=ru)
        tpc = build_tpc(default_a_bit=4, conv_a_bits=[2, 4, 8, 16], conv_w_bits=[16, 8, 4, 2],
                        fc_a_bits=[2, 4, 8], fc_w_bits=[2, 4, 8], bn_a_bits=[8, 4, 2])
        qmodel, user_info = self._run(model, datagen, tpc, ru, 4, eq_ru)
        for layer, exp_bit in exp_w_res.items():
            self._validate_weight_nbits(qmodel, layer, exp_bit)
        for layer, exp_bit in exp_a_res.items():
            self._validate_activation_nbits(qmodel, layer, exp_bit)

    @pytest.mark.parametrize('set_bops', [False, True])
    def test_all_mem_mp(self, model, datagen, set_bops):
        """ Test a combination of all memory targets with and without bops being enabled
            (bops are not a restricting factor in this test). """
        tpc = build_tpc(default_a_bit=4, conv_a_bits=[2, 4, 8, 16], conv_w_bits=[16, 8, 4, 2],
                        fc_a_bits=[2, 4, 8], fc_w_bits=[2, 4, 8], bn_a_bits=[8, 4, 2])

        def run(ru, eq_ru):
            if set_bops:
                ru.bops = 196*216*4*16 + 100*3200*8*16 + 160*200*4*8
            return self._run(model, datagen, tpc, ru, 4, eq_ru=eq_ru and not set_bops)

        max_ru = ResourceUtilization(weights_memory=3416*16/8 + 200*8/8,
                                     activation_memory=1568*8/8 + 1600*16/8,
                                     total_memory=1568*8/8 + 1600*16/8 + 3416*16/8 + 200*8/8)
        qmodel, user_info = run(max_ru, True)
        for layer, exp_bit in {'conv1': 16, 'bn1': 8, 'conv2': 16, 'fc': 8}.items():
            self._validate_activation_nbits(qmodel, layer, exp_bit)
        for layer, exp_bit in {'conv1': 16, 'conv2': 16, 'fc': 8}.items():
            self._validate_weight_nbits(qmodel, layer, exp_bit)

        # weights is the limiting target
        ru_w = ResourceUtilization(weights_memory=3416 * 2 / 8 + 200 * 2 / 8,
                                   activation_memory=100000,
                                   total_memory=100000)
        qmodel, user_info = run(ru_w, False)
        assert user_info.final_resource_utilization.weights_memory == ru_w.weights_memory
        if set_bops is False:
            # if bops is True we optimize both weights and activation. Currently, there is nothing guaranteeing that
            # a candidate (a1, 2) will have higher sensitivity than (a2>a1, 2), so we cannot know which a is selected
            for layer, exp_bit in {'conv1': 16, 'bn1': 8, 'conv2': 16, 'fc': 8}.items():
                self._validate_activation_nbits(qmodel, layer, exp_bit)
        for layer, exp_bit in {'conv1': 2, 'conv2': 2, 'fc': 2}.items():
            self._validate_weight_nbits(qmodel, layer, exp_bit)

        # activation is the limiting factor
        ru_a = ResourceUtilization(weights_memory=100000,
                                   activation_memory=1600 * 4 / 8 + 3200 * 2 / 8,
                                   total_memory=100000)
        qmodel, user_info = run(ru_a, False)
        assert user_info.final_resource_utilization.activation_memory == ru_a.activation_memory
        for layer, exp_bit in {'conv1': le(4), 'bn1': le(4), 'conv2': le(4), 'fc': 2}.items():
            self._validate_activation_nbits(qmodel, layer, exp_bit)
        if set_bops is False:
            for layer, exp_bit in {'conv1': 16, 'conv2': 16, 'fc': 8}.items():
                self._validate_weight_nbits(qmodel, layer, exp_bit)

        ru_t = ResourceUtilization(weights_memory=100000,
                                   activation_memory=100000,
                                   total_memory=1600 * 4 / 8 + 3200 * 2 / 8 + 3416 * 2 / 8 + 200 * 2 / 8)
        qmodel, user_info = run(ru_t, False)
        assert user_info.final_resource_utilization.total_memory == ru_t.total_memory
        for layer, exp_bit in {'conv1': le(4), 'bn1': le(4), 'conv2': le(4), 'fc': 2}.items():
            self._validate_activation_nbits(qmodel, layer, exp_bit)
        for layer, exp_bit in {'conv1': 2, 'conv2': 2, 'fc': 2}.items():
            self._validate_weight_nbits(qmodel, layer, exp_bit)

        # weight and activations are restricted
        ru_aw = ResourceUtilization(weights_memory=3416 * 2 / 8 + 200 * 2 / 8,
                                    activation_memory=1600 * 4 / 8 + 3200 * 2 / 8,
                                    total_memory=100000)
        qmodel, user_info = run(ru_aw, False)
        assert user_info.final_resource_utilization.weights_memory == ru_aw.weights_memory
        assert user_info.final_resource_utilization.activation_memory == ru_aw.activation_memory
        for layer, exp_bit in {'conv1': le(4), 'bn1': le(4), 'conv2': le(4), 'fc': 2}.items():
            self._validate_activation_nbits(qmodel, layer, exp_bit)
        for layer, exp_bit in {'conv1': 2, 'conv2': 2, 'fc': 2}.items():
            self._validate_weight_nbits(qmodel, layer, exp_bit)

    def test_bops_and_mem(self, model, datagen):
        """ Test a combination of bops and memory restriction, with bops being the restricting target. """
        tpc = build_tpc(default_a_bit=4, conv_a_bits=[2, 4, 8, 16], conv_w_bits=[16, 8, 4, 2],
                        fc_a_bits=[2, 4, 8], fc_w_bits=[2, 4, 8], bn_a_bits=[8, 4, 2])

        # bops limitation
        ru_b = ResourceUtilization(weights_memory=100000,
                                   activation_memory=100000,
                                   total_memory=200000,
                                   bops=196 * 216 * 4 * 2 + 100 * 3200 * 2 * 2 + 160 * 200 * 4 * 2)
        qmodel, user_info = self._run(model, datagen, tpc, ru_b, 4, False)
        assert user_info.final_resource_utilization.bops == ru_b.bops
        for layer, exp_bit in {'conv1': 16, 'bn1': 2, 'conv2': 16, 'fc': 8}.items():
            self._validate_activation_nbits(qmodel, layer, exp_bit)
        for layer, exp_bit in {'conv1': 2, 'conv2': 2, 'fc': 2}.items():
            self._validate_weight_nbits(qmodel, layer, exp_bit)

    def _run(self, model, datagen, tpc, ru, exp_default_abit, eq_ru=True, core_cfg=None):
        core_cfg = core_cfg or CoreConfig()
        qmodel, user_info = pytorch_post_training_quantization(model, datagen, target_resource_utilization=ru,
                                                               core_config=core_cfg, target_platform_capabilities=tpc)
        self._validate_sp_a_layers(qmodel, exp_default_abit)
        self._validate_ru(user_info, ru, eq_ru)
        return qmodel, user_info

    def _validate_ru(self, user_info, ru, equal):
        if equal:
            assert ru == user_info.final_resource_utilization
        else:
            assert ru.is_satisfied_by(user_info.final_resource_utilization)

    def _validate_sp_a_layers(self, qmodel, exp_nbits):
        for layer in self.sp_a_layers:
            self._validate_activation_nbits(qmodel, layer, exp_nbits)

    def _validate_activation_nbits(self, qmodel, layer_name, bits_validator: Union[Callable, int]):
        if isinstance(bits_validator, int):
            assert self.fetch_activation_holder_quantizer(qmodel, layer_name).num_bits == bits_validator, layer_name
        else:
            assert bits_validator(self.fetch_activation_holder_quantizer(qmodel, layer_name).num_bits), layer_name

    def _fetch_weight_nbit(self, qmodel, layer_name):
        layer = getattr(qmodel, layer_name)
        return layer.weights_quantizers['weight']

    def _validate_weight_nbits(self, qmodel, layer_name, exp_w_nbits):
        layer = getattr(qmodel, layer_name)
        assert layer.weights_quantizers['weight'].num_bits == exp_w_nbits, layer
        assert len(layer.weights_quantizers) == 1


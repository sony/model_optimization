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
import copy

import keras
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, Conv2DTranspose, DepthwiseConv2D, Dense, ReLU, Input, Subtract, Flatten

from model_compression_toolkit.core import QuantizationConfig
from model_compression_toolkit.core.common.graph.virtual_activation_weights_node import VirtualActivationWeightsNode, \
    VirtualSplitActivationNode, VirtualSplitWeightsNode
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization import \
    RUTarget, ResourceUtilization
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization_calculator import \
    ResourceUtilizationCalculator, TargetInclusionCriterion, BitwidthMode
from model_compression_toolkit.core.common.substitutions.apply_substitutions import substitute
from model_compression_toolkit.core.graph_prep_runner import graph_preparation_runner
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.core.keras.keras_implementation import KerasImplementation
from model_compression_toolkit.target_platform_capabilities import QuantizationMethod, AttributeQuantizationConfig, \
    OpQuantizationConfig, QuantizationConfigOptions, Signedness, OperatorsSet, OperatorSetGroup, OperatorSetNames, \
    Fusing, TargetPlatformCapabilities
from model_compression_toolkit.target_platform_capabilities.constants import KERNEL_ATTR, BIAS_ATTR
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.attach2keras import \
    AttachTpcToKeras


def build_tpc():
    default_w_cfg = AttributeQuantizationConfig(weights_quantization_method=QuantizationMethod.POWER_OF_TWO,
                                                weights_n_bits=8,
                                                weights_per_channel_threshold=True,
                                                enable_weights_quantization=True)
    default_w_nbit = 16
    default_a_nbit = 8
    default_op_cfg = OpQuantizationConfig(
        default_weight_attr_config=default_w_cfg.clone_and_edit(weights_n_bits=default_w_nbit),
        attr_weights_configs_mapping={},
        activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
        activation_n_bits=default_a_nbit,
        supported_input_activation_n_bits=[16, 8, 4, 2],
        enable_activation_quantization=True,
        quantization_preserving=False,
        fixed_scale=None,
        fixed_zero_point=None,
        simd_size=32,
        signedness=Signedness.AUTO)

    default_w_op_cfg = default_op_cfg.clone_and_edit(
        attr_weights_configs_mapping={KERNEL_ATTR: default_w_cfg, BIAS_ATTR: AttributeQuantizationConfig()}
    )

    mp_configs = []
    linear_w_min_nbit = 2
    linear_a_min_nbit = 4
    for w_nbit in [linear_w_min_nbit, linear_w_min_nbit * 2, linear_w_min_nbit * 4]:
        for a_nbit in [linear_a_min_nbit, linear_a_min_nbit * 2]:
            attr_cfg = default_w_cfg.clone_and_edit(weights_n_bits=w_nbit)
            mp_configs.append(default_w_op_cfg.clone_and_edit(
                attr_weights_configs_mapping={KERNEL_ATTR: attr_cfg, BIAS_ATTR: AttributeQuantizationConfig()},
                activation_n_bits=a_nbit
            ))
    mp_cfg_options = QuantizationConfigOptions(quantization_configurations=mp_configs,
                                               base_config=default_w_op_cfg)

    linear_ops = [OperatorsSet(name=opset, qc_options=mp_cfg_options) for opset in (OperatorSetNames.CONV,
                                                                                    OperatorSetNames.CONV_TRANSPOSE,
                                                                                    OperatorSetNames.DEPTHWISE_CONV,
                                                                                    OperatorSetNames.FULLY_CONNECTED)]

    default_cfg = QuantizationConfigOptions(quantization_configurations=[default_op_cfg])
    relu = OperatorsSet(name=OperatorSetNames.RELU, qc_options=default_cfg)

    binary_out_a_bit = 16
    binary_cfg = QuantizationConfigOptions(quantization_configurations=[default_op_cfg.clone_and_edit(activation_n_bits=binary_out_a_bit)])
    binary_ops = [OperatorsSet(name=opset, qc_options=binary_cfg) for opset in (OperatorSetNames.ADD, OperatorSetNames.SUB)]

    fusing_patterns = [Fusing(operator_groups=(OperatorSetGroup(operators_set=linear_ops), relu))]

    tpc = TargetPlatformCapabilities(default_qco=default_cfg,
                                     tpc_platform_type='test',
                                     operator_set=linear_ops + binary_ops + [relu],
                                     fusing_patterns=fusing_patterns)

    assert linear_w_min_nbit != default_w_nbit
    assert len({linear_a_min_nbit, default_a_nbit, binary_out_a_bit}) == 3
    return tpc, linear_w_min_nbit, linear_a_min_nbit, default_w_nbit, default_a_nbit, binary_out_a_bit


def data_gen():
    yield [np.random.randn(16, 16, 3)]


fw_info = DEFAULT_KERAS_INFO
fw_impl = KerasImplementation()


class TestRUIntegration:
    def test_orig_vs_virtual_sequential_graph(self):
        """ Test detailed ru computation on original and corresponding virtual graph. """
        inputs = Input(shape=(18, 18, 3))
        x = Conv2D(filters=8, kernel_size=5)(inputs)
        x = tf.add(x, np.ones((14, 8)))    # => activation with const in the composed node
        x = DepthwiseConv2D(kernel_size=3, depth_multiplier=2)(x)    # => Virtual activation in the composed node
        x = Conv2DTranspose(filters=12, kernel_size=5)(x)
        x = Flatten()(x)
        outputs = Dense(10)(x)
        model = keras.Model(inputs=inputs, outputs=outputs)

        # test the original graph
        graph, nbits = self._prepare_graph(model)
        linear_w_min_nbit, linear_a_min_nbit, default_w_nbits, default_a_nbit, binary_out_a_bit = nbits

        ru_calc = ResourceUtilizationCalculator(graph, fw_impl, fw_info)
        ru_orig, detailed_orig = ru_calc.compute_resource_utilization(TargetInclusionCriterion.AnyQuantized,
                                                                      BitwidthMode.QMinBit,
                                                                      return_detailed=True)

        exp_cuts_ru = [18 * 18 * 3 * default_a_nbit / 8,
                       (18 * 18 * 3 * default_a_nbit + 14 * 14 * 8 * linear_a_min_nbit) / 8,
                       (14 * 14 * 8 * linear_a_min_nbit + 14 * 14 * 8 * binary_out_a_bit) / 8,
                       (14 * 14 * 8 * binary_out_a_bit + 12 * 12 * 16 * linear_a_min_nbit) / 8,
                       (12 * 12 * 16 * linear_a_min_nbit + 16 * 16 * 12 * linear_a_min_nbit) / 8,
                       (16 * 16 * 12 * linear_a_min_nbit + 16 * 16 * 12 * default_a_nbit) / 8,
                       (16 * 16 * 12 * default_a_nbit + 10 * linear_a_min_nbit) / 8,
                       10 * linear_a_min_nbit / 8]
        assert self._extract_values(detailed_orig[RUTarget.ACTIVATION], sort=True) == sorted(exp_cuts_ru)

        exp_w_ru = [5 * 5 * 3 * 8 * linear_w_min_nbit / 8,
                    14 * 8 * default_w_nbits / 8,    # const
                    3 * 3 * 8 * 2 * linear_w_min_nbit / 8,
                    5 * 5 * 16 * 12 * linear_w_min_nbit / 8,
                    (16*16*12) * 10 * linear_w_min_nbit / 8]
        assert self._extract_values(detailed_orig[RUTarget.WEIGHTS]) == exp_w_ru

        exp_bops = [(5 * 5 * 3 * 8) * (14 * 14) * default_a_nbit * linear_w_min_nbit,
                    (3 * 3 * 8 * 2) * (12 * 12) * binary_out_a_bit * linear_w_min_nbit,
                    (5 * 5 * 16 * 12) * (16 * 16) * linear_a_min_nbit * linear_w_min_nbit,
                    (16 * 16 * 12) * 10 * default_a_nbit * linear_w_min_nbit]
        assert self._extract_values(detailed_orig[RUTarget.BOPS]) == exp_bops

        assert ru_orig == ResourceUtilization(activation_memory=max(exp_cuts_ru),
                                              weights_memory=sum(exp_w_ru),
                                              total_memory=max(exp_cuts_ru) + sum(exp_w_ru),
                                              bops=sum(exp_bops))

        # generate virtual graph and make sure resource utilization results are identical
        virtual_graph = substitute(copy.deepcopy(graph),
                                   fw_impl.get_substitutions_virtual_weights_activation_coupling())
        assert len(virtual_graph.nodes) == 7
        assert len([n for n in virtual_graph.nodes if isinstance(n, VirtualActivationWeightsNode)]) == 4
        assert len([n for n in virtual_graph.nodes if isinstance(n, VirtualSplitActivationNode)]) == 3

        ru_calc = ResourceUtilizationCalculator(virtual_graph, fw_impl, fw_info)
        ru_virtual, detailed_virtual = ru_calc.compute_resource_utilization(TargetInclusionCriterion.AnyQuantized,
                                                                            BitwidthMode.QMinBit,
                                                                            return_detailed=True)
        assert ru_virtual == ru_orig

        assert (self._extract_values(detailed_virtual[RUTarget.ACTIVATION], sort=True) == sorted(exp_cuts_ru))
        # virtual composed node contains both activation's const and weights' kernel
        assert (self._extract_values(detailed_virtual[RUTarget.WEIGHTS]) ==
                [exp_w_ru[0], sum(exp_w_ru[1:3]), *exp_w_ru[3:]])
        assert self._extract_values(detailed_virtual[RUTarget.BOPS]) == exp_bops

    def test_mult_output_activation(self):
        """ Tests the case when input activation has multiple outputs -> virtual weights nodes are not merged
            into VirtualActivationWeightsNode. """
        inputs = Input(shape=(16, 16, 3))
        x1 = Conv2D(filters=15, kernel_size=3, groups=3)(inputs)
        x2 = DepthwiseConv2D(kernel_size=3, depth_multiplier=5)(inputs)
        x = Subtract()([x1, x2])
        outputs = Dense(10)(x)
        model = keras.Model(inputs=inputs, outputs=outputs)

        graph, nbits = self._prepare_graph(model)
        linear_w_min_nbit, linear_a_min_nbit, default_w_nbits, default_a_nbit, binary_out_a_bit = nbits

        ru_calc = ResourceUtilizationCalculator(graph, fw_impl, fw_info)
        ru_orig, detailed_orig = ru_calc.compute_resource_utilization(TargetInclusionCriterion.AnyQuantized,
                                                                      BitwidthMode.QMinBit,
                                                                      return_detailed=True)

        exp_cuts_ru = [16*16*3*default_a_nbit/8,
                       (16*16*3*default_a_nbit + 14*14*15*linear_a_min_nbit) / 8,
                       (16 * 16 * 3 * default_a_nbit + 2 * (14 * 14 * 15 * linear_a_min_nbit)) / 8,
                       14 * 14 * 15 * (2*linear_a_min_nbit + binary_out_a_bit) / 8,
                       (14 * 14 * 15 * binary_out_a_bit + 14 * 14 * 10 * linear_a_min_nbit) / 8,
                       14 * 14 * 10 * linear_a_min_nbit / 8]

        # the order of conv and dwconv is not guaranteed, but they have same values
        exp_w_ru = [3*3*1*15*linear_w_min_nbit/8,
                    3*3*3*5*linear_w_min_nbit/8,
                    15 * 10 * linear_w_min_nbit/8]
        # bops are not computed for virtual weights nodes
        exp_bops = [(15*10)*(14*14)*binary_out_a_bit*linear_w_min_nbit]

        assert self._extract_values(detailed_orig[RUTarget.ACTIVATION], sort=True) == sorted(exp_cuts_ru)
        assert self._extract_values(detailed_orig[RUTarget.WEIGHTS]) == exp_w_ru
        assert self._extract_values(detailed_orig[RUTarget.BOPS]) == exp_bops

        virtual_graph = substitute(copy.deepcopy(graph),
                                   self.fw_impl.get_substitutions_virtual_weights_activation_coupling())
        assert len(virtual_graph.nodes) == 7
        assert len([n for n in virtual_graph.nodes if isinstance(n, VirtualActivationWeightsNode)]) == 1
        assert len([n for n in virtual_graph.nodes if isinstance(n, VirtualSplitActivationNode)]) == 3
        assert len([n for n in virtual_graph.nodes if isinstance(n, VirtualSplitWeightsNode)]) == 2

        ru_calc = ResourceUtilizationCalculator(virtual_graph, fw_impl, fw_info)
        ru_virtual, detailed_virtual = ru_calc.compute_resource_utilization(TargetInclusionCriterion.AnyQuantized,
                                                                            BitwidthMode.QMinBit,
                                                                            return_detailed=True)
        assert ru_virtual == ru_orig
        # conv and dwconv each remain as a pair of virtual W and virtual A nodes. Remaining virtual W nodes mess up the
        # cuts - but this should only add virtualW-virtualA cuts, all cuts from the original graph should stay identical
        assert not set(exp_cuts_ru) - set(detailed_virtual[RUTarget.ACTIVATION].values())
        assert self._extract_values(detailed_virtual[RUTarget.WEIGHTS]) == exp_w_ru
        assert self._extract_values(detailed_virtual[RUTarget.BOPS]) == exp_bops

    def _prepare_graph(self, model):
        tpc, *nbits = build_tpc()
        graph = graph_preparation_runner(model,
                                         data_gen,
                                         QuantizationConfig(linear_collapsing=False),
                                         fw_info=fw_info,
                                         fw_impl=fw_impl,
                                         fqc=AttachTpcToKeras().attach(tpc),
                                         mixed_precision_enable=True,
                                         running_gptq=False)
        return graph, nbits

    @staticmethod
    def _extract_values(res: dict, sort=False):
        """ Extract values for target detailed resource utilization result. """
        values = list(res.values())
        return sorted(values) if sort else values

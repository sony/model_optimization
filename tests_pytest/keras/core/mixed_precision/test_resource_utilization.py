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
import keras
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, Conv2DTranspose, DepthwiseConv2D, Dense, ReLU, Input, Subtract

from model_compression_toolkit.core import QuantizationConfig
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization import \
    RUTarget
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization_calculator import \
    ResourceUtilizationCalculator, TargetInclusionCriterion, BitwidthMode
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
    return tpc, linear_w_min_nbit, linear_a_min_nbit, default_w_nbit, default_a_nbit, binary_out_a_bit


def build_network():
    inputs = Input(shape=(16, 16, 3))
    x1 = Conv2D(filters=15, kernel_size=3, groups=3)(inputs)    # w 1*3*3*15, a 14*14*15
    x1 = ReLU()(x1)
    # TODO tpc for ADD isn't applied to +, and + is not collapsed into conv.
    # x1 = x1 + np.random.randn(14, 15)
    x1 = tf.add(x1, np.random.randn(14, 15))
    x2 = DepthwiseConv2D(kernel_size=3, depth_multiplier=5)(inputs)    # w 3*3*3*5, a 14*14*15
    x = Subtract()([x1, x2])
    x = Conv2DTranspose(filters=8, kernel_size=5)(x)    # w 5*5*15*8, a 18*18*8
    x = ReLU()(x)
    outputs = Dense(10)(x)    # w 8*10, a 10
    return keras.Model(inputs=inputs, outputs=outputs)


def data_gen():
    yield [np.random.randn(16, 16, 3)]


class TestRUIntegration:
    def test_virtual_graph(self):
        tpc, linear_w_min_nbit, linear_min_nbit, default_w_nbits, default_a_nbit, binary_out_a_bit = build_tpc()

        model = build_network()
        fw_info = DEFAULT_KERAS_INFO
        fw_impl = KerasImplementation()
        graph = graph_preparation_runner(model,
                                         data_gen,
                                         QuantizationConfig(),
                                         fw_info=fw_info,
                                         fw_impl=fw_impl,
                                         fqc=AttachTpcToKeras().attach(tpc),
                                         mixed_precision_enable=True,
                                         running_gptq=False)

        ru_calc = ResourceUtilizationCalculator(graph, fw_impl, fw_info)
        ru, detailed = ru_calc.compute_resource_utilization(TargetInclusionCriterion.AnyQuantized, BitwidthMode.QMinBit,
                                                            return_detailed=True)

        w_conv = 1 * 3 * 3 * 15    # grouped
        w_const = 14 * 15
        w_dwconv = 3 * 3 * 3 * 5
        w_conv_tr = 5 * 5 * 15 * 8
        w_dense = 8 * 10
        ru_w = list(detailed[RUTarget.WEIGHTS].values())
        assert ru_w[3:] == [w_conv_tr * linear_w_min_nbit / 8, w_dense * linear_w_min_nbit / 8]
        assert sorted(ru_w[:3]) == sorted([w_conv * linear_w_min_nbit / 8, w_const * default_w_nbits / 8, w_dwconv * linear_w_min_nbit / 8])

        assert ru.activation_memory == 14*14*15*(linear_min_nbit / 8 + 2 * binary_out_a_bit / 8)    # add - dwconv - sub
        ru_a = list(detailed[RUTarget.ACTIVATION].values())
        # the exact cuts depend on the specifics of the algorithm, we only do a sanity check
        assert 16 * 16 * 3 * default_a_nbit / 8 in ru_a    # input cut
        assert 18 * 18 * 10 * linear_min_nbit / 8 in ru_a    # output cut
        assert ru.activation_memory in ru_a    # max cut

        assert ru.total_memory == ru.activation_memory + ru.weights_memory

        ru_bops = list(detailed[RUTarget.BOPS].values())
        assert sorted(ru_bops[:2]) == sorted([w_dwconv * 14 * 14 * binary_out_a_bit * linear_w_min_nbit,
                                              w_conv * 14 * 14 * default_a_nbit * linear_w_min_nbit])
        # TODO mac computation for dense is wrong
        assert ru_bops[2:] == [w_conv_tr * 18 * 18 * binary_out_a_bit * linear_w_min_nbit,
                               w_dense * 18 * 18 * default_a_nbit * linear_w_min_nbit]

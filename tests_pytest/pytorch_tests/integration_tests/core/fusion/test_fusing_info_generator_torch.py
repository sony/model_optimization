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
from unittest.mock import Mock

import torch

from model_compression_toolkit.core import QuantizationConfig, CustomOpsetLayers
from model_compression_toolkit.core.common.fusion.fusing_info import FusingInfo
from model_compression_toolkit.target_platform_capabilities import LayerFilterParams
from tests_pytest._fw_tests_common_base.fusing.base_fusing_info_generator_test import BaseFusingInfoGeneratorTest, \
    random_activation_configs, get_activation_mp_options
from tests_pytest._test_util.graph_builder_utils import build_node
from tests_pytest.pytorch_tests.torch_test_util.torch_test_mixin import TorchFwMixin

import torch.nn as nn
import model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema as schema
from model_compression_toolkit.constants import FUSED_LAYER_PATTERN, FUSED_OP_QUANT_CONFIG


class BaseTestFusingInfoGeneratorPytorch(BaseFusingInfoGeneratorTest, TorchFwMixin):

    def _data_gen(self):
        return self.get_basic_data_gen(shapes=[(1, 3, 16, 16)])()

    def _get_qc(self):
        qc = QuantizationConfig(
            custom_tpc_opset_to_layer={"AnyAct": CustomOpsetLayers([nn.ReLU, nn.functional.relu6, nn.functional.relu, nn.SiLU, nn.Sigmoid, nn.Tanh,
                                                                    LayerFilterParams(nn.Hardtanh, min_val=0)])})
        return qc



class TestFusingConvRelu(BaseTestFusingInfoGeneratorPytorch):

    last_node_activation_nbits, qcs = random_activation_configs()

    fusing_patterns = [
        schema.Fusing(operator_groups=(
            schema.OperatorsSet(name=schema.OperatorSetNames.CONV),
            schema.OperatorsSet(name=schema.OperatorSetNames.RELU)))
    ]

    expected_fusing_patterns = [
        {FUSED_LAYER_PATTERN: [fusing_patterns[0]], FUSED_OP_QUANT_CONFIG: None}
    ]

    expected_fi = FusingInfo(
        fusing_patterns=expected_fusing_patterns,
        fusing_data={
            "FusedNode_conv1_conv2_collapsed_relu": (
                build_node(name="conv1_conv2_collapsed"),
                build_node(name="relu", qcs=qcs)
            )
        }
    )

    def _get_tpc(self, default_quant_cfg_options):
        conv = schema.OperatorsSet(name=schema.OperatorSetNames.CONV,
                                   qc_options=get_activation_mp_options(self.last_node_activation_nbits))
        relu = schema.OperatorsSet(name=schema.OperatorSetNames.RELU,
                                   qc_options=get_activation_mp_options(self.last_node_activation_nbits))
        return schema.TargetPlatformCapabilities(
            default_qco=default_quant_cfg_options,
            tpc_platform_type="test",
            operator_set=[conv, relu],
            fusing_patterns=self.fusing_patterns
        )

    def _get_model(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, kernel_size=(3, 3))
                self.conv2 = nn.Conv2d(16, 32, kernel_size=(1, 1))
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                return self.relu(x)

        return Model()

class TestFusingAnyAct(BaseTestFusingInfoGeneratorPytorch):

    last_node_activation_nbits, qcs = random_activation_configs()

    fusing_patterns = [
        schema.Fusing(operator_groups=(
            schema.OperatorsSet(name=schema.OperatorSetNames.CONV),
            schema.OperatorsSet(name="AnyAct")))
    ]

    expected_fusing_patterns = [
        {FUSED_LAYER_PATTERN: [fusing_patterns[0]], FUSED_OP_QUANT_CONFIG: None}
    ]

    expected_fi = FusingInfo(
        fusing_patterns=expected_fusing_patterns,
        fusing_data={
            "FusedNode_conv1_conv2_collapsed_tanh":
                (build_node(name="conv1_conv2_collapsed"),
                 build_node(name="tanh", qcs=qcs)),
            "FusedNode_conv3_relu":
                (build_node(name="conv3"),
                 build_node(name="relu", qcs=qcs)),
            "FusedNode_conv4_sigmoid":
                (build_node(name="conv4"),
                 build_node(name="sigmoid", qcs=qcs)),
            "FusedNode_conv5_swish":
                (build_node(name="conv5"),
                 build_node(name="swish", qcs=qcs)),
        }
    )

    def _get_tpc(self, default_quant_cfg_options):
        conv = schema.OperatorsSet(name=schema.OperatorSetNames.CONV)
        any_act = schema.OperatorsSet(name="AnyAct",
                                   qc_options=get_activation_mp_options(self.last_node_activation_nbits))
        return schema.TargetPlatformCapabilities(
            default_qco=default_quant_cfg_options,
            tpc_platform_type="test",
            operator_set=[conv, any_act],
            fusing_patterns=self.fusing_patterns
        )

    def _get_model(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3))
                self.conv2 = nn.Conv2d(32, 32, kernel_size=(1, 1))
                self.conv3 = nn.Conv2d(32, 32, kernel_size=(3, 3))
                self.conv4 = nn.Conv2d(32, 64, kernel_size=(1, 1))
                self.conv5 = nn.Conv2d(64, 64, kernel_size=(2, 2))
                self.relu = nn.ReLU()
                self.tanh = nn.Hardtanh(min_val=0)
                self.swish = nn.SiLU()
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.tanh(x)
                x = self.conv3(x)
                x = self.relu(x)
                x = self.conv4(x)
                x = self.sigmoid(x)
                x = self.conv5(x)
                return self.swish(x)

        return Model()


class TestFusingConvReLUOnly(BaseTestFusingInfoGeneratorPytorch):

    last_node_activation_nbits, qcs = random_activation_configs()

    fusing_patterns = [
        schema.Fusing(operator_groups=(
            schema.OperatorsSet(name=schema.OperatorSetNames.CONV),
            schema.OperatorsSet(name="AnyAct")))
    ]

    expected_fusing_patterns = [
        {FUSED_LAYER_PATTERN: [fusing_patterns[0]], FUSED_OP_QUANT_CONFIG: None}
    ]

    expected_fi = FusingInfo(
        fusing_patterns=expected_fusing_patterns,
        fusing_data={
            "FusedNode_conv1_conv2_collapsed_tanh":
                (build_node(name="conv1_conv2_collapsed"), build_node(name="tanh", qcs=qcs)),
            "FusedNode_conv3_relu":
                (build_node(name="conv3"), build_node(name="relu", qcs=qcs)),
            "FusedNode_conv4_sigmoid":
                (build_node(name="conv4"), build_node(name="sigmoid", qcs=qcs)),
            "FusedNode_conv5_swish":
                (build_node(name="conv5"), build_node(name="swish", qcs=qcs))
        }
    )

    def _get_tpc(self, default_quant_cfg_options):
        conv = schema.OperatorsSet(name=schema.OperatorSetNames.CONV)
        any_act = schema.OperatorsSet(name="AnyAct",
                                   qc_options=get_activation_mp_options(self.last_node_activation_nbits))
        return schema.TargetPlatformCapabilities(
            default_qco=default_quant_cfg_options,
            tpc_platform_type="test",
            operator_set=[conv, any_act],
            fusing_patterns=self.fusing_patterns
        )

    def _get_model(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3))
                self.conv2 = nn.Conv2d(32, 32, kernel_size=(1, 1))
                self.conv3 = nn.Conv2d(32, 32, kernel_size=(3, 3))
                self.conv4 = nn.Conv2d(32, 64, kernel_size=(1, 1))
                self.conv5 = nn.Conv2d(64, 64, kernel_size=(2, 2))
                self.relu = nn.ReLU()
                self.tanh = nn.Tanh()
                self.swish = nn.SiLU()
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.tanh(x)
                x = self.conv3(x)
                x = self.relu(x)
                x = self.conv4(x)
                x = self.sigmoid(x)
                x = self.conv5(x)
                return self.swish(x)

        return Model()


class TestFusingComplexPatterns(BaseTestFusingInfoGeneratorPytorch):

    last_node_activation_nbits, qcs = random_activation_configs()

    fusing_patterns = [
        schema.Fusing(operator_groups=(schema.OperatorsSet(name=schema.OperatorSetNames.CONV),
                                       schema.OperatorsSet(name=schema.OperatorSetNames.SWISH))),
        schema.Fusing(operator_groups=(schema.OperatorsSet(name=schema.OperatorSetNames.CONV),
                                       schema.OperatorsSet(name=schema.OperatorSetNames.ADD),
                                       schema.OperatorsSet(name=schema.OperatorSetNames.SWISH))),
        schema.Fusing(operator_groups=(schema.OperatorsSet(name=schema.OperatorSetNames.CONV),
                                       schema.OperatorsSet(name=schema.OperatorSetNames.SWISH),
                                       schema.OperatorsSet(name=schema.OperatorSetNames.ADD))),
        schema.Fusing(operator_groups=(schema.OperatorsSet(name=schema.OperatorSetNames.FULLY_CONNECTED),
                                       schema.OperatorsSet(name=schema.OperatorSetNames.SWISH))),
        schema.Fusing(operator_groups=(schema.OperatorsSet(name=schema.OperatorSetNames.CONV),
                                       schema.OperatorsSet(name=schema.OperatorSetNames.RELU))),
        schema.Fusing(operator_groups=(schema.OperatorsSet(name=schema.OperatorSetNames.CONV),
                                       schema.OperatorsSet(name=schema.OperatorSetNames.RELU),
                                       schema.OperatorsSet(name=schema.OperatorSetNames.ADD)))
    ]

    expected_fusing_patterns = [
        {FUSED_LAYER_PATTERN: [fusing_patterns[0]], FUSED_OP_QUANT_CONFIG: None},
        {FUSED_LAYER_PATTERN: [fusing_patterns[1]], FUSED_OP_QUANT_CONFIG: None},
        {FUSED_LAYER_PATTERN: [fusing_patterns[2]], FUSED_OP_QUANT_CONFIG: None},
        {FUSED_LAYER_PATTERN: [fusing_patterns[3]], FUSED_OP_QUANT_CONFIG: None},
        {FUSED_LAYER_PATTERN: [fusing_patterns[4]], FUSED_OP_QUANT_CONFIG: None},
        {FUSED_LAYER_PATTERN: [fusing_patterns[5]], FUSED_OP_QUANT_CONFIG: None}
    ]

    expected_fi = FusingInfo(
        fusing_patterns=expected_fusing_patterns,
        fusing_data={
            "FusedNode_conv1_swish_add": (
                build_node(name="conv1"),
                build_node(name="swish"),
                build_node(name="add", qcs=qcs)
            ),
            "FusedNode_conv2_swish_1_add_1": (
                build_node(name="conv2"),
                build_node(name="swish_1"),
                build_node(name="add_1", qcs=qcs)
            ),
            "FusedNode_conv3_relu": (
                build_node(name="conv3"),
                build_node(name="relu", qcs=qcs)
            ),
            "FusedNode_conv4_relu_1_add_2": (
                build_node(name="conv4"),
                build_node(name="relu_1"),
                build_node(name="add_2", qcs=qcs)
            ),
            "FusedNode_dense1_swish_2": (
                build_node(name="dense1"),
                build_node(name="swish_2", qcs=qcs)
            ),
            "FusedNode_dense2_swish_3": (
                build_node(name="dense2"),
                build_node(name="swish_3", qcs=qcs)
            ),
        }
    )

    def _get_tpc(self, default_quant_cfg_options):
        opsets = [
            schema.OperatorsSet(name=schema.OperatorSetNames.CONV),
            schema.OperatorsSet(name=schema.OperatorSetNames.ADD,
                                   qc_options=get_activation_mp_options(self.last_node_activation_nbits)),
            schema.OperatorsSet(name=schema.OperatorSetNames.RELU,
                                   qc_options=get_activation_mp_options(self.last_node_activation_nbits)),
            schema.OperatorsSet(name=schema.OperatorSetNames.SWISH,
                                   qc_options=get_activation_mp_options(self.last_node_activation_nbits)),
            schema.OperatorsSet(name=schema.OperatorSetNames.FULLY_CONNECTED,
                                   qc_options=get_activation_mp_options(self.last_node_activation_nbits)),
        ]
        return schema.TargetPlatformCapabilities(
            default_qco=default_quant_cfg_options,
            tpc_platform_type="test",
            operator_set=opsets,
            fusing_patterns=self.fusing_patterns
        )

    def _get_model(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 3, kernel_size=(3, 3), padding='same')
                self.conv2 = nn.Conv2d(3, 3, kernel_size=(1, 1), padding='same')
                self.conv3 = nn.Conv2d(3, 3, kernel_size=(3, 3), padding='same')
                self.conv4 = nn.Conv2d(3, 3, kernel_size=(1, 1), padding='same')
                self.conv5 = nn.Conv2d(3, 3, kernel_size=(3, 3), padding='same')
                self.conv6 = nn.Conv2d(3, 3, kernel_size=(1, 1), padding='same')
                self.relu = nn.ReLU()
                self.swish = nn.SiLU()
                self.flatten = nn.Flatten()
                self.dense1 = nn.Linear(768, out_features=16)
                self.dense2 = nn.Linear(16, out_features=16)

            def forward(self, inputs):
                x = self.conv1(inputs)
                x = self.swish(x)
                x1 = torch.add(x, inputs)
                x2 = self.conv2(x1)
                x2 = self.swish(x2)
                x2 = torch.add(x1, x2)
                x2 = self.conv3(x2)
                x2 = self.relu(x2)
                x3 = self.conv4(x2)
                x3 = self.relu(x3)
                x3 = torch.add(x3, x2)
                x3 = self.flatten(x3)
                x3 = self.dense1(x3)
                x3 = self.swish(x3)
                x3 = self.dense2(x3)
                y = self.swish(x3)
                return y

        return Model()

class TestFusingConvSwishWithMultiSuccessors(BaseTestFusingInfoGeneratorPytorch):

    last_node_activation_nbits, qcs = random_activation_configs()

    fusing_patterns = [
        schema.Fusing(operator_groups=(
            schema.OperatorsSet(name=schema.OperatorSetNames.CONV),
            schema.OperatorsSet(name=schema.OperatorSetNames.SWISH)))
    ]

    expected_fusing_patterns = [
        {FUSED_LAYER_PATTERN: [fusing_patterns[0]], FUSED_OP_QUANT_CONFIG: None}
    ]

    expected_fi = FusingInfo(
        fusing_patterns=expected_fusing_patterns,
        fusing_data={
            "FusedNode_conv1_swish": (
                build_node(name="conv1"),
                build_node(name="swish", qcs=qcs)
            )
        }
    )

    def _get_tpc(self, default_quant_cfg_options):
        conv = schema.OperatorsSet(name=schema.OperatorSetNames.CONV)
        swish = schema.OperatorsSet(name=schema.OperatorSetNames.SWISH,
                                   qc_options=get_activation_mp_options(self.last_node_activation_nbits))
        return schema.TargetPlatformCapabilities(
            default_qco=default_quant_cfg_options,
            tpc_platform_type="test",
            operator_set=[conv, swish],
            fusing_patterns=self.fusing_patterns
        )

    def _get_model(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
                self.swish = nn.SiLU()
                self.branch1 = nn.Conv2d(16, 8, kernel_size=1)
                self.branch2 = nn.Conv2d(16, 8, kernel_size=1)

            def forward(self, x):
                x = self.conv1(x)
                x = self.swish(x)
                b1 = self.branch1(x)
                b2 = self.branch2(x)
                return b1 + b2

        return Model()

class TestFusingConvReluWithMultiPredecessors(BaseTestFusingInfoGeneratorPytorch):

    last_node_activation_nbits, qcs = random_activation_configs()

    fusing_patterns = [
        schema.Fusing(operator_groups=(
            schema.OperatorsSet(name=schema.OperatorSetNames.CONV),
            schema.OperatorsSet(name=schema.OperatorSetNames.RELU)))
    ]

    expected_fusing_patterns = [
        {FUSED_LAYER_PATTERN: [fusing_patterns[0]], FUSED_OP_QUANT_CONFIG: None}
    ]

    expected_fi = FusingInfo(
        fusing_patterns=expected_fusing_patterns,
        fusing_data={
            "FusedNode_conv3_relu": (
                build_node(name="conv3"),
                build_node(name="relu", qcs=qcs)
            )
        }
    )

    def _get_tpc(self, default_quant_cfg_options):
        conv = schema.OperatorsSet(name=schema.OperatorSetNames.CONV)
        relu = schema.OperatorsSet(name=schema.OperatorSetNames.RELU,
                                   qc_options=get_activation_mp_options(self.last_node_activation_nbits))
        return schema.TargetPlatformCapabilities(
            default_qco=default_quant_cfg_options,
            tpc_platform_type="test",
            operator_set=[conv, relu],
            fusing_patterns=self.fusing_patterns
        )

    def _get_model(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
                self.conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
                self.relu = nn.ReLU()

            def forward(self, x):
                x1 = self.conv1(x)
                x2 = self.conv2(x)
                merged = x1 + x2
                x3 = self.conv3(merged)
                return self.relu(x3)

        return Model()


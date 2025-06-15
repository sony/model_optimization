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
import tensorflow as tf
from keras import Input
from tensorflow.keras import layers, Model

from model_compression_toolkit.core import QuantizationConfig, CustomOpsetLayers
from model_compression_toolkit.core.common.fusion.fusing_info import FusingInfo
from tests_pytest._fw_tests_common_base.fusing.base_fusing_info_generator_test import BaseFusingInfoGeneratorTest, \
    random_activation_configs, get_activation_mp_options
from tests_pytest._test_util.graph_builder_utils import build_node
from tests_pytest.keras_tests.keras_test_util.keras_test_mixin import KerasFwMixin
import model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema as schema
from model_compression_toolkit.constants import FUSED_LAYER_PATTERN, FUSED_OP_QUANT_CONFIG

from tensorflow.keras import backend as K


class BaseTestFusingInfoGeneratorKeras(BaseFusingInfoGeneratorTest, KerasFwMixin):

    K.clear_session() # Reset global layer naming to avoid name conflicts across tests

    def _data_gen(self):
        return self.get_basic_data_gen(shapes=[(1, 16, 16, 3)])()

    def _get_qc(self):
        qc = QuantizationConfig(
            custom_tpc_opset_to_layer={"AnyAct": CustomOpsetLayers(
                [layers.ReLU, layers.Activation, tf.nn.swish])})
        return qc


class TestFusingConvRelu(BaseTestFusingInfoGeneratorKeras):

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
        inputs = Input(shape=(None, None, 3))
        x = layers.Conv2D(16, kernel_size=(3, 3), padding='valid', name="conv1")(inputs)
        x = layers.Conv2D(32, kernel_size=(1, 1), padding='valid', name="conv2")(x)
        outputs = layers.ReLU(name="relu")(x)
        return Model(inputs=inputs, outputs=outputs)


class TestFusingAnyActKeras(BaseTestFusingInfoGeneratorKeras):

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
            "FusedNode_conv5_tf.nn.silu":
                (build_node(name="conv5"),
                 build_node(name="tf.nn.silu", qcs=qcs)),
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
        inputs = Input(shape=(32, 32, 3))
        x = layers.Conv2D(32, kernel_size=(3, 3), name="conv1")(inputs)
        x = layers.Conv2D(32, kernel_size=(1, 1), name="conv2")(x)
        x = layers.Activation("tanh", name="tanh")(x)
        x = layers.Conv2D(32, kernel_size=(3, 3), name="conv3")(x)
        x = layers.ReLU(name="relu")(x)
        x = layers.Conv2D(64, kernel_size=(1, 1), name="conv4")(x)
        x = layers.Activation("sigmoid", name="sigmoid")(x)
        x = layers.Conv2D(64, kernel_size=(2, 2), name="conv5")(x)
        outputs = layers.Activation("swish", name="tf.nn.silu")(x)
        return Model(inputs=inputs, outputs=outputs)


class TestFusingConvReLUOnlyKeras(BaseTestFusingInfoGeneratorKeras):

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
        inputs = Input(shape=(32, 32, 3))
        x = layers.Conv2D(32, kernel_size=(3, 3), name="conv1")(inputs)
        x = layers.Conv2D(32, kernel_size=(1, 1), name="conv2")(x)
        x = layers.Activation("tanh", name="tanh")(x)
        x = layers.Conv2D(32, kernel_size=(3, 3), name="conv3")(x)
        x = layers.ReLU(name="relu")(x)
        x = layers.Conv2D(64, kernel_size=(1, 1), name="conv4")(x)
        x = layers.Activation("sigmoid", name="sigmoid")(x)
        x = layers.Conv2D(64, kernel_size=(2, 2), name="conv5")(x)
        outputs = layers.Activation(tf.nn.swish, name="swish")(x)
        return Model(inputs=inputs, outputs=outputs)
class TestFusingComplexPatternsKeras(BaseTestFusingInfoGeneratorKeras):

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
                                       schema.OperatorsSet(name=schema.OperatorSetNames.ADD))),
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
            "FusedNode_conv1_swish1_add":
                (build_node(name="conv1"),
                 build_node(name="swish1"),
                 build_node(name="add", qcs=qcs)),
            "FusedNode_conv2_swish2_add_1":
                (build_node(name="conv2"),
                 build_node(name="swish2"),
                 build_node(name="add_1", qcs=qcs)),
            "FusedNode_conv3_relu":
                (build_node(name="conv3"),
                 build_node(name="relu", qcs=qcs)),
            "FusedNode_conv4_relu_1_add_2":
                (build_node(name="conv4"),
                 build_node(name="relu_1"),
                 build_node(name="add_2", qcs=qcs)),
            "FusedNode_dense1_swish3":
                (build_node(name="dense1"),
                 build_node(name="swish3", qcs=qcs)),
            "FusedNode_dense2_swish4":
                (build_node(name="dense2"),
                 build_node(name="swish4", qcs=qcs)),
        }
    )

    def _get_tpc(self, default_quant_cfg_options):
        opsets = [
            schema.OperatorsSet(name=schema.OperatorSetNames.CONV,
                                   qc_options=get_activation_mp_options(self.last_node_activation_nbits)),
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
        inputs = Input(shape=(32, 32, 3))

        x = layers.Conv2D(3, (3, 3), padding='same', name="conv1")(inputs)
        x = layers.Activation('swish', name="swish1")(x)
        x = layers.Add(name="add")([x, inputs])

        x2 = layers.Conv2D(3, (1, 1), padding='same', name="conv2")(x)
        x2 = layers.Activation('swish', name="swish2")(x2)
        x2 = layers.Add(name="add_1")([x, x2])

        x3 = layers.Conv2D(3, (3, 3), padding='same', name="conv3")(x2)
        x3 = layers.ReLU(name="relu")(x3)

        x4 = layers.Conv2D(3, (1, 1), padding='same', name="conv4")(x3)
        x4 = layers.ReLU(name="relu_1")(x4)
        x4 = layers.Add(name="add_2")([x4, x3])

        x4 = layers.Flatten()(x4)
        x4 = layers.Dense(16, name="dense1")(x4)
        x4 = layers.Activation('swish', name="swish3")(x4)

        x4 = layers.Dense(16, name="dense2")(x4)
        outputs = layers.Activation('swish', name="swish4")(x4)

        return Model(inputs=inputs, outputs=outputs)

class TestFusingConvSwishWithMultiSuccessorsKeras(BaseTestFusingInfoGeneratorKeras):

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
        conv = schema.OperatorsSet(name=schema.OperatorSetNames.CONV,
                                   qc_options=get_activation_mp_options(self.last_node_activation_nbits))
        swish = schema.OperatorsSet(name=schema.OperatorSetNames.SWISH,
                                   qc_options=get_activation_mp_options(self.last_node_activation_nbits))
        return schema.TargetPlatformCapabilities(
            default_qco=default_quant_cfg_options,
            tpc_platform_type="test",
            operator_set=[conv, swish],
            fusing_patterns=self.fusing_patterns
        )

    def _get_model(self):
        inputs = Input(shape=(32, 32, 3))
        x = layers.Conv2D(16, (3, 3), padding='same', name="conv1")(inputs)
        x = layers.Activation(tf.nn.swish, name="swish")(x)

        # Multiple successors of swish
        branch1 = layers.Conv2D(8, (1, 1), name="branch1")(x)
        branch2 = layers.Conv2D(8, (1, 1), name="branch2")(x)
        outputs = layers.Add(name="add")([branch1, branch2])
        return Model(inputs=inputs, outputs=outputs)

class TestFusingConvReluWithMultiPredecessorsKeras(BaseTestFusingInfoGeneratorKeras):

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
        inputs = Input(shape=(32, 32, 3))
        x1 = layers.Conv2D(16, (3, 3), padding='same', name="conv1")(inputs)
        x2 = layers.Conv2D(16, (3, 3), padding='same', name="conv2")(inputs)

        # Merge before relu
        merged = layers.Add(name="merge")([x1, x2])
        x = layers.Conv2D(16, (3, 3), padding='same', name="conv3")(merged)
        outputs = layers.ReLU(name="relu")(x)
        return Model(inputs=inputs, outputs=outputs)


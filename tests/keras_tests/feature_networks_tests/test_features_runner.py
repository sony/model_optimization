# Copyright 2021 Sony Semiconductors Israel, Inc. All rights reserved.
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


import logging
import unittest
import sys
from tests.keras_tests.feature_networks_tests.feature_networks.activation_scaling_relu6_test import ActivationScalingReLU6Test
from tests.keras_tests.feature_networks_tests.feature_networks.mixed_percision_test import MixedPercisionBaseTest, \
    MixedPercisionSearchTest, MixedPercisionManuallyConfiguredTest, MixedPercisionDepthwiseTest, \
    MixedPercisionSearchKPI4BitsAvgTest, MixedPercisionSearchKPI2BitsAvgTest
from tests.keras_tests.feature_networks_tests.feature_networks.multiple_inputs_node_tests import MultipleInputsNodeTests
from tests.keras_tests.feature_networks_tests.feature_networks.multiple_outputs_node_tests import MultipleOutputsNodeTests
from tests.keras_tests.feature_networks_tests.feature_networks.decompose_separable_conv_test import DecomposeSeparableConvTest
from tests.keras_tests.feature_networks_tests.feature_networks.input_scaling_test import InputScalingDenseTest, \
    InputScalingConvTest, InputScalingDWTest, InputScalingZeroPadTest
from tests.keras_tests.feature_networks_tests.feature_networks.bn_folding_test import Conv2DBNFoldingTest, \
    DepthwiseConv2DBNFoldingTest, DepthwiseConv2DBNFoldingHighMultiplierTest, Conv2DTransposeBNFoldingTest, \
    Conv2DBNConcatnFoldingTest, SeparableConv2DBNFoldingTest
from tests.keras_tests.feature_networks_tests.feature_networks.remove_upper_bound_test import RemoveUpperBoundTest
from tests.keras_tests.feature_networks_tests.feature_networks.reused_layer_mixed_precision_test import \
    ReusedLayerMixedPrecisionTest, ReusedSeparableMixedPrecisionTest
from tests.keras_tests.feature_networks_tests.feature_networks.reused_separable_test import ReusedSeparableTest
from tests.keras_tests.feature_networks_tests.feature_networks.shift_neg_activation_test import ShiftNegActivationTest
from tests.keras_tests.feature_networks_tests.feature_networks.activation_decomposition_test import ActivationDecompositionTest
from tests.keras_tests.feature_networks_tests.feature_networks.mark_activation_test import MarkActivationTest
from tests.keras_tests.feature_networks_tests.feature_networks.reused_layer_test import ReusedLayerTest
from tests.keras_tests.feature_networks_tests.feature_networks.nested_networks.nested_test import NestedTest
from tests.keras_tests.feature_networks_tests.feature_networks.nested_networks.nested_model_multiple_inputs_test import \
    NestedModelMultipleInputsTest
from tests.keras_tests.feature_networks_tests.feature_networks.nested_networks.nested_model_multiple_outputs_test import \
    NestedModelMultipleOutputsTest
from tests.keras_tests.feature_networks_tests.feature_networks.nested_networks.nested_model_unused_inputs_outputs_test import \
    NestedModelUnusedInputsOutputsTest
from tests.keras_tests.feature_networks_tests.feature_networks.multiple_output_nodes_multiple_tensors_test import \
    MultipleOutputNodesMultipleTensors
from tests.keras_tests.feature_networks_tests.feature_networks.split_concatenate_test import SplitConcatenateTest
from tests.keras_tests.feature_networks_tests.feature_networks.conv_bn_relu_residual_test import ConvBnReluResidualTest
from tests.keras_tests.feature_networks_tests.feature_networks.tanh_activation_test import TanhActivationTest
from tests.keras_tests.feature_networks_tests.feature_networks.output_in_middle_test import OutputInMiddleTest
from tests.keras_tests.feature_networks_tests.feature_networks.multiple_inputs_model_test import MultipleInputsModelTest
from tests.keras_tests.feature_networks_tests.feature_networks.scale_equalization_test import ScaleEqualizationTest
from tests.keras_tests.feature_networks_tests.feature_networks.multi_inputs_to_node_test import MultiInputsToNodeTest
from tests.keras_tests.feature_networks_tests.feature_networks.gptq_test import GradientPTQTest, \
    GradientPTQWeightsUpdateTest, GradientPTQLearnRateZeroTest
from tests.keras_tests.feature_networks_tests.feature_networks.add_same_test import AddSameTest
from tests.keras_tests.feature_networks_tests.feature_networks.split_test import SplitTest
from tests.keras_tests.feature_networks_tests.feature_networks.network_editor.node_filter_test import NameFilterTest, \
    ScopeFilterTest, TypeFilterTest
import tensorflow as tf

layers = tf.keras.layers


class FeatureNetworkTest(unittest.TestCase):

    def test_remove_upper_bound(self):
        RemoveUpperBoundTest(self).run_test()

    def test_reused_separable_mixed_precision(self):
        ReusedSeparableMixedPrecisionTest(self).run_test()

    def test_reused_layer_mixed_precision(self):
        ReusedLayerMixedPrecisionTest(self).run_test()

    def test_reuse_separable(self):
        ReusedSeparableTest(self).run_test()

    def test_mixed_precision_search_kpi_2bits_avg(self):
        MixedPercisionSearchKPI2BitsAvgTest(self).run_test()

    def test_mixed_precision_search_kpi_4bits_avg(self):
        MixedPercisionSearchKPI4BitsAvgTest(self).run_test()

    def test_mixed_precision_search(self):
        MixedPercisionSearchTest(self).run_test()

    def test_mixed_precision_dw(self):
        MixedPercisionDepthwiseTest(self).run_test()

    def test_name_filter(self):
        NameFilterTest(self).run_test()

    def test_scope_filter(self):
        ScopeFilterTest(self).run_test()

    def test_type_filter(self):
        TypeFilterTest(self).run_test()

    def test_split(self):
        SplitTest(self).run_test()

    def test_add_same(self):
        AddSameTest(self).run_test()

    def test_scale_equalization(self):
        ScaleEqualizationTest(self, first_op2d=layers.Dense(30), second_op2d=layers.Dense(30)).run_test()
        ScaleEqualizationTest(self, first_op2d=layers.Dense(30), second_op2d=layers.Dense(30),
                              mid_activation=True).run_test()
        ScaleEqualizationTest(self, first_op2d=layers.Dense(30), second_op2d=layers.Conv2D(3, 4)).run_test()
        ScaleEqualizationTest(self, first_op2d=layers.Dense(30), second_op2d=layers.Conv2D(3, 4),
                              mid_activation=True).run_test()
        ScaleEqualizationTest(self, first_op2d=layers.DepthwiseConv2D(11), second_op2d=layers.Conv2D(3, 4)).run_test()
        ScaleEqualizationTest(self, first_op2d=layers.DepthwiseConv2D(11), second_op2d=layers.Conv2D(3, 4),
                              mid_activation=True).run_test()
        ScaleEqualizationTest(self, first_op2d=layers.Dense(30), second_op2d=layers.DepthwiseConv2D(3)).run_test()
        ScaleEqualizationTest(self, first_op2d=layers.Dense(30), second_op2d=layers.DepthwiseConv2D(3),
                              mid_activation=True).run_test()

        ScaleEqualizationTest(self, first_op2d=layers.Dense(30), second_op2d=layers.Dense(30),
                              second_op2d_zero_pad=True).run_test()
        ScaleEqualizationTest(self, first_op2d=layers.Dense(30), second_op2d=layers.Dense(30),
                              mid_activation=True, second_op2d_zero_pad=True).run_test()
        ScaleEqualizationTest(self, first_op2d=layers.Dense(30), second_op2d=layers.Conv2D(3, 4),
                              second_op2d_zero_pad=True).run_test()
        ScaleEqualizationTest(self, first_op2d=layers.Dense(30), second_op2d=layers.Conv2D(3, 4),
                              mid_activation=True, second_op2d_zero_pad=True).run_test()
        ScaleEqualizationTest(self, first_op2d=layers.DepthwiseConv2D(11), second_op2d=layers.Conv2D(3, 4),
                              second_op2d_zero_pad=True).run_test()
        ScaleEqualizationTest(self, first_op2d=layers.DepthwiseConv2D(11), second_op2d=layers.Conv2D(3, 4),
                              mid_activation=True, second_op2d_zero_pad=True).run_test()
        ScaleEqualizationTest(self, first_op2d=layers.Dense(30), second_op2d=layers.DepthwiseConv2D(3),
                              second_op2d_zero_pad=True).run_test()
        ScaleEqualizationTest(self, first_op2d=layers.Dense(30), second_op2d=layers.DepthwiseConv2D(3),
                              mid_activation=True, second_op2d_zero_pad=True).run_test()

    def test_multiple_inputs_model(self):
        MultipleInputsModelTest(self).run_test()

    def test_output_in_middle(self):
        OutputInMiddleTest(self).run_test()

    def test_tanh_activation(self):
        TanhActivationTest(self).run_test()

    def test_conv_bn_relu_residual(self):
        ConvBnReluResidualTest(self).run_test()

    def test_split_concat(self):
        SplitConcatenateTest(self).run_test()

    def test_multiple_output_nodes_multiple_tensors(self):
        MultipleOutputNodesMultipleTensors(self).run_test()

    def test_reused_layer(self):
        ReusedLayerTest(self).run_test()

    def test_nested_model_multiple_inputs(self):
        NestedModelMultipleInputsTest(self).run_test()

    def test_nested_model_multiple_outputs(self):
        NestedModelMultipleOutputsTest(self).run_test()

    def test_nested_model_unused_inputs_outputs(self):
        NestedModelUnusedInputsOutputsTest(self).run_test()

    def test_nested_model(self):
        NestedTest(self).run_test()
        NestedTest(self, is_inner_functional=False).run_test()

    def test_shift_neg_activation_conv2d(self):
        ShiftNegActivationTest(self, linear_op_to_test=layers.Conv2D(3, 4)).run_test()
        ShiftNegActivationTest(self, linear_op_to_test=layers.Conv2D(3, 4, strides=3)).run_test()
        ShiftNegActivationTest(self, linear_op_to_test=layers.Conv2D(3, (3, 4), strides=2)).run_test()

    def test_shift_neg_activation_conv2d_pad_same(self):
        ShiftNegActivationTest(self, linear_op_to_test=layers.Conv2D(3, (5, 7), padding='same')).run_test()
        ShiftNegActivationTest(self, linear_op_to_test=layers.Conv2D(3, (5, 7), padding='same', strides=3)).run_test()
        ShiftNegActivationTest(self, linear_op_to_test=layers.Conv2D(3, (7, 5), padding='same', strides=4)).run_test()
        ShiftNegActivationTest(self, linear_op_to_test=layers.Conv2D(3, (8, 10), padding='same', strides=5)).run_test()
        ShiftNegActivationTest(self, linear_op_to_test=layers.Conv2D(3, (10, 8), padding='same', strides=6)).run_test()
        ShiftNegActivationTest(self,
                               linear_op_to_test=layers.Conv2D(3, (5, 7), padding='same', strides=(4, 6))).run_test()
        ShiftNegActivationTest(self,
                               linear_op_to_test=layers.Conv2D(3, (7, 5), padding='same', strides=(6, 4))).run_test()
        ShiftNegActivationTest(self,
                               linear_op_to_test=layers.Conv2D(3, (8, 10), padding='same', strides=(5, 7))).run_test()
        ShiftNegActivationTest(self,
                               linear_op_to_test=layers.Conv2D(3, (10, 8), padding='same', strides=(7, 5))).run_test()

    def test_shift_neg_activation_pad_conv2d(self):
        ShiftNegActivationTest(self, linear_op_to_test=layers.Conv2D(3, (5, 7)), use_pad_layer=True).run_test()
        ShiftNegActivationTest(self, linear_op_to_test=layers.Conv2D(3, (5, 7), strides=3),
                               use_pad_layer=True).run_test()
        ShiftNegActivationTest(self, linear_op_to_test=layers.Conv2D(3, (7, 5), strides=4),
                               use_pad_layer=True).run_test()
        ShiftNegActivationTest(self, linear_op_to_test=layers.Conv2D(3, (8, 10), strides=5),
                               use_pad_layer=True).run_test()
        ShiftNegActivationTest(self, linear_op_to_test=layers.Conv2D(3, (10, 8), strides=6),
                               use_pad_layer=True).run_test()

    def test_shift_neg_activation_dense(self):
        ShiftNegActivationTest(self, linear_op_to_test=layers.Dense(3)).run_test()
        ShiftNegActivationTest(self, linear_op_to_test=layers.Dense(4)).run_test()

    def test_shift_neg_activation_pad_dense(self):
        ShiftNegActivationTest(self, linear_op_to_test=layers.Dense(3), use_pad_layer=True).run_test()
        ShiftNegActivationTest(self, linear_op_to_test=layers.Dense(4), use_pad_layer=True).run_test()

    def test_shift_neg_activation_depthwise(self):
        ShiftNegActivationTest(self, linear_op_to_test=layers.DepthwiseConv2D((4, 5))).run_test()
        ShiftNegActivationTest(self, linear_op_to_test=layers.DepthwiseConv2D(5, strides=3)).run_test()
        ShiftNegActivationTest(self, linear_op_to_test=layers.DepthwiseConv2D((5, 4), strides=4)).run_test()

    def test_shift_neg_activation_depthwise_pad_same(self):
        ShiftNegActivationTest(self, linear_op_to_test=layers.DepthwiseConv2D(depth_multiplier=1, kernel_size=(7, 8),
                                                                              padding='same', strides=5)).run_test()
        ShiftNegActivationTest(self, linear_op_to_test=layers.DepthwiseConv2D(depth_multiplier=3, kernel_size=(8, 7),
                                                                              padding='same', strides=6)).run_test()
        ShiftNegActivationTest(self, linear_op_to_test=layers.DepthwiseConv2D(kernel_size=4, padding='same')).run_test()

    def test_shift_neg_activation_pad_depthwise(self):
        ShiftNegActivationTest(self, linear_op_to_test=layers.DepthwiseConv2D((4, 5)), use_pad_layer=True).run_test()
        ShiftNegActivationTest(self, linear_op_to_test=layers.DepthwiseConv2D(5, strides=3),
                               use_pad_layer=True).run_test()
        ShiftNegActivationTest(self, linear_op_to_test=layers.DepthwiseConv2D((5, 4), strides=4),
                               use_pad_layer=True).run_test()

    def test_activation_decomposition(self):
        ActivationDecompositionTest(self, activation_function='swish').run_test()
        ActivationDecompositionTest(self, activation_function='relu').run_test()
        ActivationDecompositionTest(self, activation_function='tanh').run_test()
        ActivationDecompositionTest(self, activation_function='softmax').run_test()

    def test_mark_activation(self):
        MarkActivationTest(self, layers.Dense, layers.PReLU()).run_test()
        MarkActivationTest(self, layers.Conv2D, layers.ReLU()).run_test()
        MarkActivationTest(self, layers.Conv2DTranspose, layers.Activation('sigmoid')).run_test()
        MarkActivationTest(self, layers.DepthwiseConv2D, layers.Activation('softmax')).run_test()

    def test_conv2d_bn_concant(self):
        Conv2DBNConcatnFoldingTest(self).run_test()

    def test_activation_scaling_relu6(self):
        ActivationScalingReLU6Test(self).run_test()

    def test_multiple_inputs_node(self):
        MultipleInputsNodeTests(self).run_test()

    def test_multiple_outputs_node(self):
        MultipleOutputsNodeTests(self).run_test()

    def test_conv2dbn_folding(self):
        Conv2DBNFoldingTest(self).run_test()

    def test_separableconv2dbn_folding(self):
        SeparableConv2DBNFoldingTest(self).run_test()

    def test_dwbn_folding(self):
        DepthwiseConv2DBNFoldingTest(self).run_test()

    def test_dwbn_high_multipler_folding(self):
        DepthwiseConv2DBNFoldingHighMultiplierTest(self).run_test()

    def test_conv2dtransposebn_folding(self):
        Conv2DTransposeBNFoldingTest(self).run_test()

    def test_decompose_separable_conv(self):
        DecomposeSeparableConvTest(self).run_test()

    def test_decompose_separable_conv_high_multiplier(self):
        DecomposeSeparableConvTest(self, depth=2).run_test()

    def test_input_scale(self):
        InputScalingDenseTest(self).run_test()
        InputScalingConvTest(self).run_test()
        InputScalingDWTest(self).run_test()
        InputScalingZeroPadTest(self).run_test()

    def test_multi_input_to_node(self):
        MultiInputsToNodeTest(self).run_test()

    def test_gptq(self):
        GradientPTQTest(self).run_test()
        GradientPTQWeightsUpdateTest(self).run_test()
        GradientPTQLearnRateZeroTest(self).run_test()


if __name__ == '__main__':
    unittest.main()

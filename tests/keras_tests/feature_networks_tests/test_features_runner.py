# Copyright 2021 Sony Semiconductor Israel, Inc. All rights reserved.
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


import unittest

import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import distance_metrics
from tensorflow.keras.layers import PReLU, ELU

from model_compression_toolkit.core import QuantizationErrorMethod
from model_compression_toolkit.core.common.mixed_precision.distance_weighting import MpDistanceWeighting
from model_compression_toolkit.target_platform_capabilities.target_platform import QuantizationMethod
from model_compression_toolkit.gptq import RoundingType
from model_compression_toolkit.target_platform_capabilities import constants as C
from tests.keras_tests.feature_networks_tests.feature_networks.activation_decomposition_test import \
    ActivationDecompositionTest
from tests.keras_tests.feature_networks_tests.feature_networks.activation_relu_bound_to_power_of_2_test import \
    ReLUBoundToPOTNetTest
from tests.keras_tests.feature_networks_tests.feature_networks.add_same_test import AddSameTest
from tests.keras_tests.feature_networks_tests.feature_networks.bias_correction_dw_test import \
    BiasCorrectionDepthwiseTest
from tests.keras_tests.feature_networks_tests.feature_networks.bn_attributes_quantization_test import \
    BNAttributesQuantization
from tests.keras_tests.feature_networks_tests.feature_networks.bn_folding_test import Conv2DBNFoldingTest, \
    DepthwiseConv2DBNFoldingTest, DepthwiseConv2DBNFoldingHighMultiplierTest, Conv2DTransposeBNFoldingTest, \
    Conv2DBNConcatFoldingTest, SeparableConv2DBNFoldingTest, BNForwardFoldingTest
from tests.keras_tests.feature_networks_tests.feature_networks.compute_max_cut_test import ComputeMaxCutTest
from tests.keras_tests.feature_networks_tests.feature_networks.conv_bn_relu_residual_test import ConvBnReluResidualTest
from tests.keras_tests.feature_networks_tests.feature_networks.decompose_separable_conv_test import \
    DecomposeSeparableConvTest
from tests.keras_tests.feature_networks_tests.feature_networks.experimental_exporter_test import \
    ExportableModelTest
from tests.keras_tests.feature_networks_tests.feature_networks.gptq.gptq_conv import \
    GradientPTQLearnRateZeroConvGroupDilationTest, GradientPTQWeightsUpdateConvGroupDilationTest
from tests.keras_tests.feature_networks_tests.feature_networks.gptq.gptq_test import GradientPTQTest, \
    GradientPTQWeightsUpdateTest, GradientPTQLearnRateZeroTest, GradientPTQWeightedLossTest, \
    GradientPTQNoTempLearningTest, GradientPTQWithDepthwiseTest
from tests.keras_tests.feature_networks_tests.feature_networks.input_scaling_test import InputScalingDenseTest, \
    InputScalingConvTest, InputScalingDWTest, InputScalingZeroPadTest
from tests.keras_tests.feature_networks_tests.feature_networks.linear_collapsing_test import TwoConv2DCollapsingTest, \
    ThreeConv2DCollapsingTest, FourConv2DCollapsingTest, SixConv2DCollapsingTest, Op2DAddConstCollapsingTest
from tests.keras_tests.feature_networks_tests.feature_networks.lut_quantizer import LUTWeightsQuantizerTest, \
    LUTActivationQuantizerTest
from tests.keras_tests.feature_networks_tests.feature_networks.mixed_precision.requires_mixed_precision_test import \
    RequiresMixedPrecision, RequiresMixedPrecisionWeights
from tests.keras_tests.feature_networks_tests.feature_networks.mixed_precision_bops_test import \
    MixedPrecisionBopsBasicTest, MixedPrecisionBopsAllWeightsLayersTest, MixedPrecisionWeightsOnlyBopsTest, \
    MixedPrecisionActivationOnlyBopsTest, MixedPrecisionBopsAndWeightsUtilizationTest, MixedPrecisionBopsAndActivationUtilizationTest, \
    MixedPrecisionBopsAndTotalUtilizationTest, MixedPrecisionBopsWeightsActivationUtilizationTest, \
    MixedPrecisionBopsMultipleOutEdgesTest
from tests.keras_tests.feature_networks_tests.feature_networks.mixed_precision_tests import \
    MixedPrecisionActivationSearchTest, MixedPrecisionActivationSearch4BitsAvgTest, \
    MixedPrecisionActivationSearch2BitsAvgTest, MixedPrecisionActivationDepthwiseTest, \
    MixedPrecisionActivationSplitLayerTest, MixedPrecisionActivationOnlyWeightsDisabledTest, \
    MixedPrecisionActivationOnlyTest, MixedPrecisionActivationDepthwise4BitTest, MixedPrecisionActivationAddLayerTest, \
    MixedPrecisionActivationMultipleInputsTest, MixedPrecisionTotalMemoryUtilizationSearchTest, \
    MixedPrecisionMultipleResourcesTightUtilizationSearchTest, MixedPrecisionReducedTotalMemorySearchTest
from tests.keras_tests.feature_networks_tests.feature_networks.multi_head_attention_test import MultiHeadAttentionTest
from tests.keras_tests.feature_networks_tests.feature_networks.multi_inputs_to_node_test import MultiInputsToNodeTest
from tests.keras_tests.feature_networks_tests.feature_networks.multiple_inputs_model_test import MultipleInputsModelTest
from tests.keras_tests.feature_networks_tests.feature_networks.multiple_inputs_node_tests import MultipleInputsNodeTests
from tests.keras_tests.feature_networks_tests.feature_networks.multiple_output_nodes_multiple_tensors_test import \
    MultipleOutputNodesMultipleTensors
from tests.keras_tests.feature_networks_tests.feature_networks.multiple_outputs_node_tests import \
    MultipleOutputsNodeTests
from tests.keras_tests.feature_networks_tests.feature_networks.nested_networks.nested_model_multiple_inputs_test import \
    NestedModelMultipleInputsTest
from tests.keras_tests.feature_networks_tests.feature_networks.nested_networks.nested_model_multiple_outputs_test import \
    NestedModelMultipleOutputsTest
from tests.keras_tests.feature_networks_tests.feature_networks.nested_networks.nested_model_unused_inputs_outputs_test import \
    NestedModelUnusedInputsOutputsTest
from tests.keras_tests.feature_networks_tests.feature_networks.nested_networks.nested_test import NestedTest
from tests.keras_tests.feature_networks_tests.feature_networks.network_editor.change_qc_attr_test import \
    ChangeFinalWeightQCAttrTest, ChangeFinalActivationQCAttrTest
from tests.keras_tests.feature_networks_tests.feature_networks.network_editor.edit_error_method_test import \
    EditActivationErrorMethod
from tests.keras_tests.feature_networks_tests.feature_networks.network_editor.edit_qc_test import \
    ChangeCandidatesWeightsQuantConfigAttrTest, ChangeCandidatesActivationQCAttrTest, \
    ChangeFinalsWeightsQuantConfigAttrTest, ChangeFinalsActivationQCAttrTest, \
    ChangeCandidatesActivationQuantizationMethodQCAttrTest, ChangeCandidatesWeightsQuantizationMethodQCAttrTest, \
    ChangeFinalsActivationQuantizationMethodQCAttrTest, ChangeFinalsWeightsQuantizationMethodQCAttrTest
from tests.keras_tests.feature_networks_tests.feature_networks.network_editor.node_filter_test import NameFilterTest, \
    ScopeFilterTest, TypeFilterTest
from tests.keras_tests.feature_networks_tests.feature_networks.output_in_middle_test import OutputInMiddleTest
from tests.keras_tests.feature_networks_tests.feature_networks.per_tensor_weight_quantization_test import \
    PerTensorWeightQuantizationTest
from tests.keras_tests.feature_networks_tests.feature_networks.qat.qat_test import QATWrappersTest, \
    QuantizationAwareTrainingQuantizersTest, QATWrappersMixedPrecisionCfgTest, \
    QuantizationAwareTrainingQuantizerHolderTest
from tests.keras_tests.feature_networks_tests.feature_networks.relu_replacement_test import ReluReplacementTest, \
    SingleReluReplacementTest, ReluReplacementWithAddBiasTest
from tests.keras_tests.feature_networks_tests.feature_networks.remove_identity_test import RemoveIdentityTest
from tests.keras_tests.feature_networks_tests.feature_networks.residual_collapsing_test import ResidualCollapsingTest1, \
    ResidualCollapsingTest2
from tests.keras_tests.feature_networks_tests.feature_networks.reused_layer_mixed_precision_test import \
    ReusedLayerMixedPrecisionTest, ReusedSeparableMixedPrecisionTest
from tests.keras_tests.feature_networks_tests.feature_networks.reused_layer_test import ReusedLayerTest
from tests.keras_tests.feature_networks_tests.feature_networks.reused_separable_test import ReusedSeparableTest
from tests.keras_tests.feature_networks_tests.feature_networks.scale_equalization_test import ScaleEqualizationTest
from tests.keras_tests.feature_networks_tests.feature_networks.second_moment_correction_test import \
    DepthwiseConv2DSecondMomentTest, Conv2DSecondMomentTest, Conv2DTSecondMomentTest, \
    ValueSecondMomentTest, POTSecondMomentTest, NoBNSecondMomentTest, ReusedConvSecondMomentTest, \
    UniformSecondMomentTest
from tests.keras_tests.feature_networks_tests.feature_networks.shift_neg_activation_test import ShiftNegActivationTest, \
    ShiftNegActivationPostAddTest
from tests.keras_tests.feature_networks_tests.feature_networks.softmax_shift_test import SoftmaxShiftTest
from tests.keras_tests.feature_networks_tests.feature_networks.split_concatenate_test import SplitConcatenateTest
from tests.keras_tests.feature_networks_tests.feature_networks.split_conv_bug_test import SplitConvBugTest
from tests.keras_tests.feature_networks_tests.feature_networks.symmetric_threshold_selection_activation_test import \
    SymmetricThresholdSelectionActivationTest, SymmetricThresholdSelectionBoundedActivationTest
from tests.keras_tests.feature_networks_tests.feature_networks.test_depthwise_conv2d_replacement import \
    DwConv2dReplacementTest
from tests.keras_tests.feature_networks_tests.feature_networks.test_kmeans_quantizer import \
    KmeansQuantizerTestManyClasses
from tests.keras_tests.feature_networks_tests.feature_networks.uniform_range_selection_activation_test import \
    UniformRangeSelectionActivationTest, UniformRangeSelectionBoundedActivationTest
from tests.keras_tests.feature_networks_tests.feature_networks.weights_mixed_precision_tests import \
    MixedPrecisionSearch4BitsAvgTest, MixedPrecisionSearch2BitsAvgTest, MixedPrecisionActivationDisabled, \
    MixedPrecisionWithHessianScoresTest, MixedPrecisionSearchTest, \
    MixedPrecisionSearchPartWeightsLayersTest, MixedPrecisionDepthwiseTest, MixedPrecisionSearchLastLayerDistanceTest, \
    MixedPrecisionSearchActivationNonConfNodesTest, MixedPrecisionSearchTotalMemoryNonConfNodesTest, \
    MixedPrecisionCombinedNMSTest
from tests.keras_tests.feature_networks_tests.feature_networks.matmul_substitution_test import MatmulToDenseSubstitutionTest
from tests.keras_tests.feature_networks_tests.feature_networks.metadata_test import MetadataTest
from tests.keras_tests.feature_networks_tests.feature_networks.tpc_test import TpcTest
from tests.keras_tests.feature_networks_tests.feature_networks.const_representation_test import ConstRepresentationTest, \
    ConstRepresentationMultiInputTest, ConstRepresentationMatMulTest
from tests.keras_tests.feature_networks_tests.feature_networks.concatination_threshold_update import ConcatThresholdtest
from tests.keras_tests.feature_networks_tests.feature_networks.const_quantization_test import ConstQuantizationTest, \
    AdvancedConstQuantizationTest
from model_compression_toolkit.qat.common.qat_config import TrainingMethod

layers = tf.keras.layers


class FeatureNetworkTest(unittest.TestCase):

    def test_compute_max_cut(self):
        ComputeMaxCutTest(self).run_test()

    def test_remove_identity(self):
        RemoveIdentityTest(self).run_test()

    def test_per_tensor_weight_quantization(self):
        PerTensorWeightQuantizationTest(self).run_test()
    
    def test_single_relu_replacement(self):
        SingleReluReplacementTest(self).run_test()

    def test_relu_replacement(self):
        ReluReplacementTest(self).run_test()

    def test_relu_add_bias_replacement(self):
        ReluReplacementWithAddBiasTest(self).run_test()

    def test_depthwise_conv2d_replacement(self):
        DwConv2dReplacementTest(self).run_test()

    def test_edit_error_method(self):
        EditActivationErrorMethod(self).run_test()

    def test_change_qc_attr(self):
        ChangeFinalWeightQCAttrTest(self).run_test()
        ChangeFinalActivationQCAttrTest(self).run_test()

    def test_edit_candidate_qc(self):
        ChangeCandidatesWeightsQuantConfigAttrTest(self).run_test()
        ChangeCandidatesActivationQCAttrTest(self).run_test()
        ChangeCandidatesActivationQuantizationMethodQCAttrTest(self).run_test()
        ChangeCandidatesWeightsQuantizationMethodQCAttrTest(self).run_test()

    def test_edit_final_qc(self):
        ChangeFinalsWeightsQuantConfigAttrTest(self).run_test()
        ChangeFinalsActivationQCAttrTest(self).run_test()
        ChangeFinalsActivationQuantizationMethodQCAttrTest(self).run_test()
        ChangeFinalsWeightsQuantizationMethodQCAttrTest(self).run_test()

    def test_bias_correction_dw(self):
        BiasCorrectionDepthwiseTest(self).run_test()

    def test_lut_quantizer(self):
        LUTWeightsQuantizerTest(self).run_test()
        LUTWeightsQuantizerTest(self, is_symmetric=True).run_test()
        LUTActivationQuantizerTest(self).run_test()

    def test_kmeans_quantizer(self):
        # In this test we have weights with less unique values than the number of clusters
        KmeansQuantizerTestManyClasses(self, QuantizationMethod.LUT_POT_QUANTIZER,
                                       weights_n_bits=7).run_test()

    def test_reused_separable_mixed_precision(self):
        ReusedSeparableMixedPrecisionTest(self).run_test()

    def test_reused_layer_mixed_precision(self):
        ReusedLayerMixedPrecisionTest(self).run_test()

    def test_reuse_separable(self):
        ReusedSeparableTest(self).run_test()

    def test_mixed_precision_search_2bits_avg(self):
        MixedPrecisionSearch2BitsAvgTest(self).run_test()

    def test_mixed_precision_search_4bits_avg(self):
        MixedPrecisionSearch4BitsAvgTest(self).run_test()

    def test_mixed_precision_search_4bits_avg_nms(self):
        MixedPrecisionCombinedNMSTest(self).run_test()

    def test_mixed_precision_search(self):
        MixedPrecisionSearchTest(self, distance_metric=MpDistanceWeighting.AVG).run_test()
        MixedPrecisionSearchTest(self, distance_metric=MpDistanceWeighting.LAST_LAYER).run_test()
        MixedPrecisionWithHessianScoresTest(self, distance_metric=MpDistanceWeighting.AVG).run_test()

    def test_requires_mixed_recision(self):
        RequiresMixedPrecisionWeights(self, weights_memory=True).run_test()
        RequiresMixedPrecision(self,activation_memory=True).run_test()
        RequiresMixedPrecision(self, total_memory=True).run_test()
        RequiresMixedPrecision(self, bops=True).run_test()
        RequiresMixedPrecision(self).run_test()

    def test_mixed_precision_for_part_weights_layers(self):
        MixedPrecisionSearchPartWeightsLayersTest(self).run_test()

    def test_mixed_precision_activation_disabled(self):
        MixedPrecisionActivationDisabled(self).run_test()

    def test_mixed_precision_dw(self):
        MixedPrecisionDepthwiseTest(self).run_test()

    def test_mixed_precision_search_with_last_layer_distance(self):
        MixedPrecisionSearchLastLayerDistanceTest(self).run_test()

    def test_mixed_precision_search_activation_non_conf_nodes(self):
        MixedPrecisionSearchActivationNonConfNodesTest(self).run_test()

    def test_mixed_precision_search_total_non_conf_nodes(self):
        MixedPrecisionSearchTotalMemoryNonConfNodesTest(self).run_test()

    def test_mixed_precision_activation_search(self):
        MixedPrecisionActivationSearchTest(self).run_test()

    def test_mixed_precision_activation_only(self):
        MixedPrecisionActivationOnlyTest(self).run_test()

    def test_mixed_precision_activation_only_weights_disabled(self):
        MixedPrecisionActivationOnlyWeightsDisabledTest(self).run_test()

    def test_mixed_precision_activation_search_4bits_avg(self):
        MixedPrecisionActivationSearch4BitsAvgTest(self).run_test()

    def test_mixed_precision_activation_search_2bits_avg(self):
        MixedPrecisionActivationSearch2BitsAvgTest(self).run_test()

    def test_mixed_precision_activation_dw(self):
        MixedPrecisionActivationDepthwiseTest(self).run_test()

    def test_mixed_precision_activation_dw_4bit(self):
        MixedPrecisionActivationDepthwise4BitTest(self).run_test()

    def test_mixed_precision_activation_add(self):
        MixedPrecisionActivationAddLayerTest(self).run_test()

    def test_mixed_precision_activation_split(self):
        MixedPrecisionActivationSplitLayerTest(self).run_test()

    def test_mixed_precision_activation_multiple_inputs(self):
        MixedPrecisionActivationMultipleInputsTest(self).run_test()

    def test_mixed_precision_total_memory_utilization(self):
        MixedPrecisionTotalMemoryUtilizationSearchTest(self).run_test()

    def test_mixed_precision_multiple_resources_tight_utilization(self):
        MixedPrecisionMultipleResourcesTightUtilizationSearchTest(self).run_test()

    def test_mixed_precision_reduced_total_memory(self):
        MixedPrecisionReducedTotalMemorySearchTest(self).run_test()

    def test_mixed_precision_bops_utilization(self):
        MixedPrecisionBopsBasicTest(self).run_test()
        MixedPrecisionBopsAllWeightsLayersTest(self).run_test()
        MixedPrecisionWeightsOnlyBopsTest(self).run_test()
        MixedPrecisionActivationOnlyBopsTest(self).run_test()
        MixedPrecisionBopsAndWeightsUtilizationTest(self).run_test()
        MixedPrecisionBopsAndActivationUtilizationTest(self).run_test()
        MixedPrecisionBopsAndTotalUtilizationTest(self).run_test()
        MixedPrecisionBopsWeightsActivationUtilizationTest(self).run_test()
        MixedPrecisionBopsMultipleOutEdgesTest(self).run_test()

    def test_name_filter(self):
        NameFilterTest(self).run_test()

    def test_scope_filter(self):
        ScopeFilterTest(self).run_test()

    def test_type_filter(self):
        TypeFilterTest(self).run_test()

    def test_add_same(self):
        AddSameTest(self).run_test()

    def test_scale_equalization(self):
        ScaleEqualizationTest(self, first_op2d=layers.Conv2D(3, 4), second_op2d=layers.Conv2D(3, 4)).run_test()
        ScaleEqualizationTest(self, first_op2d=layers.Conv2D(3, 4), second_op2d=layers.Conv2DTranspose(3, 4)).run_test()
        ScaleEqualizationTest(self, first_op2d=layers.Conv2D(3, 4), second_op2d=layers.DepthwiseConv2D(3, 4)).run_test()

        ScaleEqualizationTest(self, first_op2d=layers.Conv2D(3, 4), second_op2d=layers.Conv2D(3, 4),
                              zero_pad=True).run_test()
        ScaleEqualizationTest(self, first_op2d=layers.Conv2D(3, 4), second_op2d=layers.Conv2DTranspose(3, 4),
                              zero_pad=True).run_test()
        ScaleEqualizationTest(self, first_op2d=layers.Conv2D(3, 4), second_op2d=layers.DepthwiseConv2D(3, 4),
                              zero_pad=True).run_test()

        ScaleEqualizationTest(self, first_op2d=layers.Conv2D(3, 4), second_op2d=layers.Conv2D(3, 4),
                              act_node=tf.nn.relu).run_test()
        ScaleEqualizationTest(self, first_op2d=layers.Conv2D(3, 4), second_op2d=layers.Conv2DTranspose(3, 4),
                              act_node=tf.nn.relu).run_test()
        ScaleEqualizationTest(self, first_op2d=layers.Conv2D(3, 4), second_op2d=layers.DepthwiseConv2D(3, 4),
                              act_node=tf.nn.relu).run_test()

        ScaleEqualizationTest(self, first_op2d=layers.Conv2D(3, 4), second_op2d=layers.Conv2D(3, 4),
                              act_node=tf.nn.relu, zero_pad=True).run_test()
        ScaleEqualizationTest(self, first_op2d=layers.Conv2D(3, 4), second_op2d=layers.Conv2DTranspose(3, 4),
                              act_node=tf.nn.relu, zero_pad=True).run_test()
        ScaleEqualizationTest(self, first_op2d=layers.Conv2D(3, 4), second_op2d=layers.DepthwiseConv2D(3, 4),
                              act_node=tf.nn.relu, zero_pad=True).run_test()

        ScaleEqualizationTest(self, first_op2d=layers.Conv2DTranspose(3, 4), second_op2d=layers.Conv2D(3, 4)).run_test()
        ScaleEqualizationTest(self, first_op2d=layers.DepthwiseConv2D(3, 4), second_op2d=layers.Conv2DTranspose(3, 4)
                              ).run_test()

        ScaleEqualizationTest(self, first_op2d=layers.DepthwiseConv2D(3, 4), second_op2d=layers.Conv2D(3, 4),
                              zero_pad=True).run_test()
        ScaleEqualizationTest(self, first_op2d=layers.Conv2DTranspose(3, 4), second_op2d=layers.Conv2DTranspose(3, 4),
                              zero_pad=True).run_test()

        ScaleEqualizationTest(self, first_op2d=layers.DepthwiseConv2D(3, 4), second_op2d=layers.Conv2D(3, 4),
                              act_node=tf.nn.relu).run_test()
        ScaleEqualizationTest(self, first_op2d=layers.Conv2DTranspose(3, 4), second_op2d=layers.DepthwiseConv2D(3, 4),
                              act_node=tf.nn.relu).run_test()

        ScaleEqualizationTest(self, first_op2d=layers.Conv2DTranspose(3, 4), second_op2d=layers.Conv2DTranspose(3, 4),
                              act_node=tf.nn.relu, zero_pad=True).run_test()
        ScaleEqualizationTest(self, first_op2d=layers.DepthwiseConv2D(3, 4), second_op2d=layers.DepthwiseConv2D(3, 4),
                              act_node=tf.nn.relu, zero_pad=True).run_test()

    def test_multiple_inputs_model(self):
        MultipleInputsModelTest(self).run_test()

    def test_output_in_middle(self):
        OutputInMiddleTest(self).run_test()

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
        ShiftNegActivationTest(self, linear_op_to_test=layers.Conv2D(3, 4),
                               activation_op_to_test=layers.Activation('swish'), param_search=True).run_test()
        ShiftNegActivationTest(self, linear_op_to_test=layers.Conv2D(3, 4, strides=3),
                               activation_op_to_test=layers.Activation('swish')).run_test()
        ShiftNegActivationTest(self, linear_op_to_test=layers.Conv2D(3, (3, 4), strides=2),
                               activation_op_to_test=layers.Activation('swish')).run_test()

    def test_shift_neg_activation_conv2d_pad_same(self):
        ShiftNegActivationTest(self, linear_op_to_test=layers.Conv2D(3, (5, 7), padding='same'),
                               activation_op_to_test=layers.Activation('swish')).run_test()
        ShiftNegActivationTest(self, linear_op_to_test=layers.Conv2D(3, (5, 7), padding='same', strides=3),
                               activation_op_to_test=layers.Activation('swish')).run_test()
        ShiftNegActivationTest(self, linear_op_to_test=layers.Conv2D(3, (7, 5), padding='same', strides=4),
                               activation_op_to_test=layers.Activation('swish')).run_test()
        ShiftNegActivationTest(self, linear_op_to_test=layers.Conv2D(3, (8, 10), padding='same', strides=5),
                               activation_op_to_test=layers.Activation('swish')).run_test()
        ShiftNegActivationTest(self, linear_op_to_test=layers.Conv2D(3, (10, 8), padding='same', strides=6),
                               activation_op_to_test=layers.Activation('swish')).run_test()
        ShiftNegActivationTest(self, linear_op_to_test=layers.Conv2D(3, (5, 7), padding='same', strides=(4, 6)),
                               activation_op_to_test=layers.Activation('swish')).run_test()
        ShiftNegActivationTest(self, linear_op_to_test=layers.Conv2D(3, (7, 5), padding='same', strides=(6, 4)),
                               activation_op_to_test=layers.Activation('swish')).run_test()
        ShiftNegActivationTest(self, linear_op_to_test=layers.Conv2D(3, (8, 10), padding='same', strides=(5, 7)),
                               activation_op_to_test=layers.Activation('swish')).run_test()
        ShiftNegActivationTest(self, linear_op_to_test=layers.Conv2D(3, (10, 8), padding='same', strides=(7, 5)),
                               activation_op_to_test=layers.Activation('swish')).run_test()

    def test_shift_neg_activation_pad_conv2d(self):
        ShiftNegActivationTest(self, linear_op_to_test=layers.Conv2D(3, (5, 7)),
                               activation_op_to_test=layers.Activation('swish'),
                               use_pad_layer=True).run_test()
        ShiftNegActivationTest(self, linear_op_to_test=layers.Conv2D(3, (5, 7), strides=3),
                               activation_op_to_test=layers.Activation('swish'),
                               use_pad_layer=True).run_test()
        ShiftNegActivationTest(self, linear_op_to_test=layers.Conv2D(3, (7, 5), strides=4),
                               activation_op_to_test=layers.Activation('swish'),
                               use_pad_layer=True, param_search=True).run_test()
        ShiftNegActivationTest(self, linear_op_to_test=layers.Conv2D(3, (8, 10), strides=5),
                               activation_op_to_test=layers.Activation('swish'),
                               use_pad_layer=True).run_test()
        ShiftNegActivationTest(self, linear_op_to_test=layers.Conv2D(3, (10, 8), strides=6),
                               activation_op_to_test=layers.Activation('swish'),
                               use_pad_layer=True).run_test()

    def test_shift_neg_activation_dense(self):
        ShiftNegActivationTest(self, linear_op_to_test=layers.Dense(3),
                               activation_op_to_test=tf.nn.leaky_relu, input_shape=(8, 3)).run_test()
        ShiftNegActivationTest(self, linear_op_to_test=layers.Dense(4),
                               activation_op_to_test=layers.ReLU(negative_slope=0.99),
                               bypass_op_list=[layers.GlobalAveragePooling2D()]).run_test()

    def test_shift_neg_activation_pad_dense(self):
        ShiftNegActivationTest(self, linear_op_to_test=layers.Dense(3),
                               activation_op_to_test=PReLU(alpha_initializer='ones'), use_pad_layer=True).run_test()
        ShiftNegActivationTest(self, linear_op_to_test=layers.Dense(4),
                               activation_op_to_test=ELU(), use_pad_layer=True).run_test()

    def test_shift_neg_activation_depthwise(self):
        ShiftNegActivationTest(self, linear_op_to_test=layers.DepthwiseConv2D((4, 5)),
                               activation_op_to_test=tf.nn.silu).run_test()
        ShiftNegActivationTest(self, linear_op_to_test=layers.DepthwiseConv2D(5, strides=3),
                               activation_op_to_test=tf.nn.elu).run_test()
        ShiftNegActivationTest(self, linear_op_to_test=layers.DepthwiseConv2D((5, 4), strides=4),
                               activation_op_to_test=tf.nn.leaky_relu).run_test()

    def test_shift_neg_activation_depthwise_pad_same(self):
        ShiftNegActivationTest(self, linear_op_to_test=layers.DepthwiseConv2D(depth_multiplier=1, kernel_size=(7, 8),
                                                                              padding='same', strides=5),
                               activation_op_to_test=layers.Activation('swish')).run_test()
        ShiftNegActivationTest(self, linear_op_to_test=layers.DepthwiseConv2D(depth_multiplier=3, kernel_size=(8, 7),
                                                                              padding='same', strides=6),
                               activation_op_to_test=layers.Activation('gelu')).run_test()
        ShiftNegActivationTest(self, linear_op_to_test=layers.DepthwiseConv2D(kernel_size=4, padding='same'),
                               activation_op_to_test=layers.Activation('selu')).run_test()

    def test_shift_neg_activation_pad_depthwise(self):
        ShiftNegActivationTest(self, linear_op_to_test=layers.DepthwiseConv2D((4, 5)),
                               activation_op_to_test=tf.nn.gelu, use_pad_layer=True).run_test()
        ShiftNegActivationTest(self, linear_op_to_test=layers.DepthwiseConv2D(5, strides=3),
                               activation_op_to_test=tf.nn.selu, use_pad_layer=True).run_test()
        ShiftNegActivationTest(self, linear_op_to_test=layers.DepthwiseConv2D((5, 4), strides=4),
                               activation_op_to_test=tf.nn.swish, use_pad_layer=True).run_test()

    def test_shift_neg_activation_post_add(self):
        ShiftNegActivationPostAddTest(self,
                                      linear_op_to_test=layers.Conv2D(3, 4),
                                      activation_op_to_test=layers.Activation('swish'),
                                      post_add_nbits=7).run_test()

    def test_activation_decomposition(self):
        ActivationDecompositionTest(self, activation_function='swish').run_test()
        ActivationDecompositionTest(self, activation_function='relu').run_test()
        ActivationDecompositionTest(self, activation_function='tanh').run_test()
        ActivationDecompositionTest(self, activation_function='softmax').run_test()

    def test_experimental_exporter(self):
        ExportableModelTest(self).run_test()

    def test_matmul_dense_substitution(self):
        MatmulToDenseSubstitutionTest(self).run_test()

    def test_conv2d_bn_concat(self):
        Conv2DBNConcatFoldingTest(self).run_test()

    def test_activation_scaling_relu6(self):
        ReLUBoundToPOTNetTest(self).run_test()

    def test_layer_activation_softmax_shift(self):
        SoftmaxShiftTest(self, layers.Dense(20, activation='softmax'), None).run_test()

    def test_layer_softmax_shift(self):
        SoftmaxShiftTest(self, layers.Dense(20), layers.Softmax()).run_test()

    def test_function_softmax_shift(self):
        SoftmaxShiftTest(self, layers.Dense(20), tf.nn.softmax).run_test()

    def test_multiple_inputs_node(self):
        MultipleInputsNodeTests(self).run_test()

    def test_multiple_outputs_node(self):
        MultipleOutputsNodeTests(self).run_test()

    def test_conv2dbn_folding(self):
        Conv2DBNFoldingTest(self).run_test()

    def test_bn_forward_folding(self):
        BNForwardFoldingTest(self, layers.Conv2D(2, 1, bias_initializer='glorot_uniform'), True, is_dwconv=True).run_test()
        BNForwardFoldingTest(self, layers.DepthwiseConv2D(1, bias_initializer='glorot_uniform'), True, is_dwconv=True).run_test()
        BNForwardFoldingTest(self, layers.Conv2DTranspose(2, 1, bias_initializer='glorot_uniform'), True, is_dwconv=True).run_test()
        BNForwardFoldingTest(self, layers.Conv2D(2, 2), False, is_dwconv=True).run_test()
        BNForwardFoldingTest(self, layers.DepthwiseConv2D((3, 1)), False, is_dwconv=True).run_test()
        BNForwardFoldingTest(self, layers.Conv2DTranspose(2, (1, 3)), False, is_dwconv=True).run_test()
        BNForwardFoldingTest(self, layers.Conv2D(2, 1, bias_initializer='glorot_uniform'),
                             True, add_bn=True, is_dwconv=True).run_test()

        BNForwardFoldingTest(self, layers.Conv2D(2, 1, bias_initializer='glorot_uniform'), True).run_test()
        BNForwardFoldingTest(self, layers.DepthwiseConv2D(1, bias_initializer='glorot_uniform'), True).run_test()
        BNForwardFoldingTest(self, layers.Conv2DTranspose(2, 1, bias_initializer='glorot_uniform'), True).run_test()
        BNForwardFoldingTest(self, layers.Conv2D(2, 2), False).run_test()
        BNForwardFoldingTest(self, layers.DepthwiseConv2D((3, 1)), False).run_test()
        BNForwardFoldingTest(self, layers.Conv2DTranspose(2, (1, 3)), False).run_test()
        BNForwardFoldingTest(self, layers.Conv2D(2, 1, bias_initializer='glorot_uniform'),
                             True, add_bn=True).run_test()

    def test_residual_collapsing(self):
        ResidualCollapsingTest1(self).run_test()
        ResidualCollapsingTest2(self).run_test()

    def test_separableconv2dbn_folding(self):
        SeparableConv2DBNFoldingTest(self).run_test()

    def test_dwbn_folding(self):
        DepthwiseConv2DBNFoldingTest(self).run_test()

    def test_dwbn_high_multipler_folding(self):
        DepthwiseConv2DBNFoldingHighMultiplierTest(self).run_test()

    def test_conv2dtransposebn_folding(self):
        Conv2DTransposeBNFoldingTest(self).run_test()

    def test_linear_collapsing(self):
        TwoConv2DCollapsingTest(self).run_test()
        ThreeConv2DCollapsingTest(self).run_test()
        FourConv2DCollapsingTest(self).run_test()
        SixConv2DCollapsingTest(self).run_test()
        Op2DAddConstCollapsingTest(self).run_test()

    def test_const_quantization(self):
        c = (np.ones((32, 32, 16)) + np.random.random((32, 32, 16))).astype(np.float32)
        for func in [tf.add, tf.multiply, tf.subtract, tf.divide, tf.truediv]:
            for qmethod in [QuantizationErrorMethod.MSE, QuantizationErrorMethod.NOCLIPPING]:
                ConstQuantizationTest(self, func, c, qmethod=qmethod).run_test()
                ConstQuantizationTest(self, func, c, input_reverse_order=True, qmethod=qmethod).run_test()
                ConstQuantizationTest(self, func, c, input_reverse_order=True, use_kwargs=True, qmethod=qmethod).run_test()
                ConstQuantizationTest(self, func, c, use_kwargs=True, qmethod=qmethod).run_test()
                ConstQuantizationTest(self, func, 2.45, qmethod=qmethod).run_test()
                ConstQuantizationTest(self, func, 5.1, input_reverse_order=True, qmethod=qmethod).run_test()

        AdvancedConstQuantizationTest(self).run_test()

    def test_const_representation(self):
        c = (np.ones((16,)) + np.random.random((16,))).astype(np.float32)
        for func in [tf.add, tf.multiply, tf.subtract, tf.divide, tf.truediv, tf.pow]:
            ConstRepresentationTest(self, func, c).run_test()
            ConstRepresentationTest(self, func, c, input_reverse_order=True).run_test()
            ConstRepresentationTest(self, func, c, input_reverse_order=True, use_kwargs=True).run_test()
            ConstRepresentationTest(self, func, c, use_kwargs=True).run_test()
            ConstRepresentationTest(self, func, 2.45).run_test()
            ConstRepresentationTest(self, func, 5.1, input_reverse_order=True).run_test()

        # tf.matmul test
        ConstRepresentationMatMulTest(self).run_test()

        c = (np.ones((16,)) + np.random.random((16,))).astype(np.float32).reshape((1, -1))
        for func in [layers.Add(), layers.Multiply(), layers.Subtract()]:
            ConstRepresentationTest(self, func, c, is_list_input=True).run_test()
            ConstRepresentationTest(self, func, c, input_reverse_order=True, is_list_input=True).run_test()
            ConstRepresentationTest(self, func, c, input_reverse_order=True, use_kwargs=True, is_list_input=True).run_test()
            ConstRepresentationTest(self, func, c, use_kwargs=True, is_list_input=True).run_test()

        ConstRepresentationMultiInputTest(self).run_test()

    def test_second_moment(self):
        DepthwiseConv2DSecondMomentTest(self).run_test()
        # DepthwiseConv2DWithMultiplierSecondMomentTest(self).run_test()
        Conv2DSecondMomentTest(self).run_test()
        Conv2DTSecondMomentTest(self).run_test()
        ValueSecondMomentTest(self).run_test()
        POTSecondMomentTest(self).run_test()
        UniformSecondMomentTest(self).run_test()
        ReusedConvSecondMomentTest(self).run_test()
        NoBNSecondMomentTest(self).run_test()

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
        # This call removes the effect of @tf.function decoration and executes the decorated function eagerly, which
        # enabled tracing for code coverage.
        tf.config.run_functions_eagerly(True)
        GradientPTQTest(self).run_test()
        GradientPTQTest(self, per_channel=True).run_test()
        GradientPTQTest(self, per_channel=True, hessian_weights=False).run_test()
        GradientPTQTest(self, per_channel=True, log_norm_weights=False).run_test()
        GradientPTQWeightsUpdateTest(self).run_test()
        GradientPTQLearnRateZeroTest(self).run_test()
        GradientPTQWeightedLossTest(self).run_test()
        GradientPTQTest(self,
                        quant_method=QuantizationMethod.UNIFORM,
                        rounding_type=RoundingType.SoftQuantizer,
                        per_channel=False,
                        quantization_parameter_learning=False).run_test()
        GradientPTQTest(self,
                        quant_method=QuantizationMethod.UNIFORM,
                        rounding_type=RoundingType.SoftQuantizer,
                        per_channel=True,
                        quantization_parameter_learning=False).run_test()
        GradientPTQLearnRateZeroTest(self,
                                     quant_method=QuantizationMethod.UNIFORM,
                                     rounding_type=RoundingType.SoftQuantizer,
                                     quantization_parameter_learning=False).run_test()
        GradientPTQTest(self,
                        rounding_type=RoundingType.SoftQuantizer,
                        per_channel=False).run_test()
        GradientPTQTest(self,
                        rounding_type=RoundingType.SoftQuantizer,
                        per_channel=True).run_test()
        GradientPTQTest(self,
                        rounding_type=RoundingType.SoftQuantizer,
                        per_channel=True, hessian_weights=True, log_norm_weights=True, scaled_log_norm=True).run_test()
        GradientPTQWeightedLossTest(self,
                                    rounding_type=RoundingType.SoftQuantizer,
                                    per_channel=True, hessian_weights=True, log_norm_weights=True,
                                    scaled_log_norm=True).run_test()
        GradientPTQNoTempLearningTest(self,
                                      rounding_type=RoundingType.SoftQuantizer).run_test()
        GradientPTQWeightsUpdateTest(self,
                                     rounding_type=RoundingType.SoftQuantizer).run_test()
        GradientPTQLearnRateZeroTest(self,
                                     rounding_type=RoundingType.SoftQuantizer).run_test()
        GradientPTQWithDepthwiseTest(self,
                                     rounding_type=RoundingType.SoftQuantizer).run_test()

        tf.config.run_functions_eagerly(False)

    # TODO: reuven - new experimental facade needs to be tested regardless the exporter.
    # def test_gptq_new_exporter(self):
    #     self.test_gptq(experimental_exporter=True)

    # Comment out due to problem in Tensorflow 2.8
    # def test_gptq_conv_group(self):
    #     GradientPTQLearnRateZeroConvGroupTest(self).run_test()
    #     GradientPTQWeightsUpdateConvGroupTest(self).run_test()


    def test_gptq_conv_group_dilation(self):
        GradientPTQLearnRateZeroConvGroupDilationTest(self).run_test()
        GradientPTQWeightsUpdateConvGroupDilationTest(self).run_test()
        GradientPTQLearnRateZeroConvGroupDilationTest(self, rounding_type=RoundingType.SoftQuantizer).run_test()
        GradientPTQWeightsUpdateConvGroupDilationTest(self, rounding_type=RoundingType.SoftQuantizer).run_test()

    def test_split_conv_bug(self):
        SplitConvBugTest(self).run_test()

    def test_symmetric_threshold_selection_activation(self):
        SymmetricThresholdSelectionActivationTest(self, QuantizationErrorMethod.NOCLIPPING).run_test()
        SymmetricThresholdSelectionActivationTest(self, QuantizationErrorMethod.MSE).run_test()
        SymmetricThresholdSelectionActivationTest(self, QuantizationErrorMethod.MAE).run_test()
        SymmetricThresholdSelectionActivationTest(self, QuantizationErrorMethod.LP).run_test()
        SymmetricThresholdSelectionActivationTest(self, QuantizationErrorMethod.KL).run_test()

    def test_symmetric_threshold_selection_softmax_activation(self):
        SymmetricThresholdSelectionBoundedActivationTest(self, QuantizationErrorMethod.NOCLIPPING).run_test()
        SymmetricThresholdSelectionBoundedActivationTest(self, QuantizationErrorMethod.MSE).run_test()
        SymmetricThresholdSelectionBoundedActivationTest(self, QuantizationErrorMethod.MAE).run_test()
        SymmetricThresholdSelectionBoundedActivationTest(self, QuantizationErrorMethod.LP).run_test()
        SymmetricThresholdSelectionBoundedActivationTest(self, QuantizationErrorMethod.KL).run_test()

    def test_uniform_range_selection_activation(self):
        UniformRangeSelectionActivationTest(self, QuantizationErrorMethod.NOCLIPPING).run_test()
        UniformRangeSelectionActivationTest(self, QuantizationErrorMethod.MSE).run_test()
        UniformRangeSelectionActivationTest(self, QuantizationErrorMethod.MAE).run_test()
        UniformRangeSelectionActivationTest(self, QuantizationErrorMethod.LP).run_test()
        UniformRangeSelectionActivationTest(self, QuantizationErrorMethod.KL).run_test()

    def test_uniform_range_selection_softmax_activation(self):
        UniformRangeSelectionBoundedActivationTest(self, QuantizationErrorMethod.NOCLIPPING).run_test()
        UniformRangeSelectionBoundedActivationTest(self, QuantizationErrorMethod.MSE).run_test()
        UniformRangeSelectionBoundedActivationTest(self, QuantizationErrorMethod.MAE).run_test()
        UniformRangeSelectionBoundedActivationTest(self, QuantizationErrorMethod.LP).run_test()
        UniformRangeSelectionBoundedActivationTest(self, QuantizationErrorMethod.KL).run_test()

    def test_multi_head_attention(self):
        q_seq_len, kv_seq_len = 5, 6
        q_dim, k_dim, v_dim = 11, 12, 13
        num_heads, qk_proj_dim, v_proj_dim = 3, 4, 7
        attention_axes = [1, 3]
        num_iterations = 9
        for separate_key_value in [False, True]:
            MultiHeadAttentionTest(self, [(q_seq_len, q_dim),
                                          (kv_seq_len, k_dim),
                                          (kv_seq_len, v_dim)],
                                   num_heads, qk_proj_dim, v_proj_dim, None,
                                   separate_key_value=separate_key_value, output_dim=15).run_test()
            input_shapes = [(2, num_iterations, q_seq_len, q_dim),
                            (2, num_iterations, kv_seq_len, k_dim),
                            (2, num_iterations, kv_seq_len, v_dim)]
            MultiHeadAttentionTest(self, input_shapes,
                                   num_heads, qk_proj_dim, v_proj_dim, attention_axes,
                                   separate_key_value=separate_key_value, output_dim=14).run_test()
            MultiHeadAttentionTest(self, input_shapes,
                                   num_heads, qk_proj_dim, v_proj_dim, attention_axes,
                                   separate_key_value=separate_key_value, output_dim=None).run_test()
            MultiHeadAttentionTest(self, input_shapes,
                                   num_heads, qk_proj_dim, v_proj_dim, None,
                                   separate_key_value=separate_key_value, output_dim=14).run_test()

    def test_qat(self):
        QATWrappersTest(self, layers.Conv2D(3, 4, activation='relu'), test_loading=True).run_test()
        QATWrappersTest(self, layers.Conv2D(3, 4, activation='relu'), test_loading=True, per_channel=False).run_test()
        QATWrappersTest(self, layers.Conv2D(3, 4, activation='relu'),
                        weights_quantization_method=QuantizationMethod.UNIFORM,
                        activation_quantization_method=QuantizationMethod.SYMMETRIC).run_test()
        QATWrappersTest(self, layers.Dense(3, activation='relu'),
                        weights_quantization_method=QuantizationMethod.UNIFORM,
                        activation_quantization_method=QuantizationMethod.UNIFORM,
                        test_loading=True, per_channel=False).run_test()
        QATWrappersTest(self, layers.Dense(3, activation='relu')).run_test()
        QATWrappersTest(self, layers.Conv2DTranspose(3, 4, activation='relu'), test_loading=True,
                        weights_quantization_method=QuantizationMethod.SYMMETRIC,
                        activation_quantization_method=QuantizationMethod.SYMMETRIC).run_test()
        QATWrappersTest(self, layers.Conv2DTranspose(3, 4, activation='relu')).run_test()
        QATWrappersTest(self, layers.DepthwiseConv2D(3, 4, activation='relu'),
                        weights_quantization_method=QuantizationMethod.SYMMETRIC,
                        activation_quantization_method=QuantizationMethod.SYMMETRIC,
                        training_method=TrainingMethod.LSQ).run_test()
        QATWrappersTest(self, layers.Conv2D(3, 4, activation='relu'),
                        weights_quantization_method=QuantizationMethod.UNIFORM,
                        activation_quantization_method=QuantizationMethod.UNIFORM,
                        training_method=TrainingMethod.LSQ).run_test()
        QATWrappersTest(self, layers.DepthwiseConv2D(3, 4, activation='relu'),
                        weights_quantization_method=QuantizationMethod.UNIFORM,
                        activation_quantization_method=QuantizationMethod.UNIFORM,
                        training_method=TrainingMethod.LSQ).run_test()
        QATWrappersTest(self, layers.Dense(3, activation='relu'),
                        weights_quantization_method=QuantizationMethod.POWER_OF_TWO,
                        activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
                        training_method=TrainingMethod.LSQ).run_test()
        QuantizationAwareTrainingQuantizersTest(self).run_test()
        QuantizationAwareTrainingQuantizerHolderTest(self).run_test()
        QATWrappersMixedPrecisionCfgTest(self).run_test()
        QATWrappersMixedPrecisionCfgTest(self, ru_weights=17920 * 4 / 8, ru_activation=5408 * 4 / 8, expected_mp_cfg=[0, 4, 1, 1]).run_test()

    def test_bn_attributes_quantization(self):
        BNAttributesQuantization(self, quantize_linear=False).run_test()
        BNAttributesQuantization(self, quantize_linear=True).run_test()

    def test_concat_threshold(self):
        ConcatThresholdtest(self).run_test()

    def test_metadata(self):
        MetadataTest(self).run_test()

    def test_keras_tpcs(self):
        TpcTest(f'{C.IMX500_TP_MODEL}.v1', self).run_test()
        TpcTest(f'{C.IMX500_TP_MODEL}.v1_lut', self).run_test()
        TpcTest(f'{C.IMX500_TP_MODEL}.v1_pot', self).run_test()
        TpcTest(f'{C.IMX500_TP_MODEL}.v2', self).run_test()
        TpcTest(f'{C.IMX500_TP_MODEL}.v2_lut', self).run_test()
        TpcTest(f'{C.IMX500_TP_MODEL}.v3', self).run_test()
        TpcTest(f'{C.IMX500_TP_MODEL}.v3_lut', self).run_test()
        TpcTest(f'{C.TFLITE_TP_MODEL}.v1', self).run_test()
        TpcTest(f'{C.QNNPACK_TP_MODEL}.v1', self).run_test()


if __name__ == '__main__':
    unittest.main()

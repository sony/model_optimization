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

import tensorflow as tf
from tensorflow.keras.layers import PReLU, ELU

import model_compression_toolkit as mct
from model_compression_toolkit import QuantizationErrorMethod
from tests.keras_tests.feature_networks_tests.feature_networks.activation_decomposition_test import \
    ActivationDecompositionTest
from tests.keras_tests.feature_networks_tests.feature_networks.activation_relu_bound_to_power_of_2_test import \
    ReLUBoundToPOTNetTest
from tests.keras_tests.feature_networks_tests.feature_networks.add_same_test import AddSameTest
from tests.keras_tests.feature_networks_tests.feature_networks.bias_correction_dw_test import \
    BiasCorrectionDepthwiseTest
from tests.keras_tests.feature_networks_tests.feature_networks.bn_folding_test import Conv2DBNFoldingTest, \
    DepthwiseConv2DBNFoldingTest, DepthwiseConv2DBNFoldingHighMultiplierTest, Conv2DTransposeBNFoldingTest, \
    Conv2DBNConcatnFoldingTest, SeparableConv2DBNFoldingTest
from tests.keras_tests.feature_networks_tests.feature_networks.conv_bn_relu_residual_test import ConvBnReluResidualTest
from tests.keras_tests.feature_networks_tests.feature_networks.decompose_separable_conv_test import \
    DecomposeSeparableConvTest
from tests.keras_tests.feature_networks_tests.feature_networks.experimental_exporter_test import \
    ExperimentalExporterTest
from tests.keras_tests.feature_networks_tests.feature_networks.gptq.gptq_conv import \
    GradientPTQLearnRateZeroConvGroupDilationTest, GradientPTQWeightsUpdateConvGroupDilationTest
from tests.keras_tests.feature_networks_tests.feature_networks.gptq.gptq_test import GradientPTQTest, \
    GradientPTQWeightsUpdateTest, GradientPTQLearnRateZeroTest, GradientPTQWeightedLossTest, \
    GradientPTQNoTempLearningTest
from tests.keras_tests.feature_networks_tests.feature_networks.input_scaling_test import InputScalingDenseTest, \
    InputScalingConvTest, InputScalingDWTest, InputScalingZeroPadTest
from tests.keras_tests.feature_networks_tests.feature_networks.linear_collapsing_test import TwoConv2DCollapsingTest, \
    ThreeConv2DCollapsingTest, FourConv2DCollapsingTest, SixConv2DCollapsingTest
from tests.keras_tests.feature_networks_tests.feature_networks.lut_quantizer import LUTWeightsQuantizerTest, \
    LUTActivationQuantizerTest
from tests.keras_tests.feature_networks_tests.feature_networks.mixed_precision_bops_test import \
    MixedPrecisionBopsBasicTest, MixedPrecisionBopsAllWeightsLayersTest, MixedPrecisionWeightsOnlyBopsTest, \
    MixedPrecisionActivationOnlyBopsTest, MixedPrecisionBopsAndWeightsKPITest, MixedPrecisionBopsAndActivationKPITest, \
    MixedPrecisionBopsAndTotalKPITest, MixedPrecisionBopsWeightsActivationKPITest, \
    MixedPrecisionBopsMultipleOutEdgesTest
from tests.keras_tests.feature_networks_tests.feature_networks.mixed_precision_tests import \
    MixedPrecisionActivationSearchTest, MixedPrecisionActivationSearchKPI4BitsAvgTest, \
    MixedPrecisionActivationSearchKPI2BitsAvgTest, MixedPrecisionActivationDepthwiseTest, \
    MixedPrecisionActivationSplitLayerTest, MixedPrecisionActivationOnlyWeightsDisabledTest, \
    MixedPrecisionActivationOnlyTest, MixedPrecisionActivationDepthwise4BitTest, MixedPrecisionActivationAddLayerTest, \
    MixedPrecisionActivationMultipleInputsTest, MixedPrecisionTotalKPISearchTest, \
    MixedPrecisionMultipleKPIsTightSearchTest, MixedPrecisionReducedTotalKPISearchTest
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
from tests.keras_tests.feature_networks_tests.feature_networks.qat.qat_test import QuantizationAwareTrainingTest, \
    QuantizationAwareTrainingQuantizersTest
from tests.keras_tests.feature_networks_tests.feature_networks.relu_replacement_test import ReluReplacementTest, \
    SingleReluReplacementTest, ReluReplacementWithAddBiasTest
from tests.keras_tests.feature_networks_tests.feature_networks.remove_upper_bound_test import RemoveUpperBoundTest
from tests.keras_tests.feature_networks_tests.feature_networks.residual_collapsing_test import ResidualCollapsingTest1, \
    ResidualCollapsingTest2
from tests.keras_tests.feature_networks_tests.feature_networks.reused_layer_mixed_precision_test import \
    ReusedLayerMixedPrecisionTest, ReusedSeparableMixedPrecisionTest
from tests.keras_tests.feature_networks_tests.feature_networks.reused_layer_test import ReusedLayerTest
from tests.keras_tests.feature_networks_tests.feature_networks.reused_separable_test import ReusedSeparableTest
from tests.keras_tests.feature_networks_tests.feature_networks.scale_equalization_test import ScaleEqualizationTest
from tests.keras_tests.feature_networks_tests.feature_networks.second_moment_correction_test import \
    DepthwiseConv2DSecondMomentTest, Conv2DSecondMomentTest, Conv2DTSecondMomentTest, \
    ValueSecondMomentTest
from tests.keras_tests.feature_networks_tests.feature_networks.shift_neg_activation_test import ShiftNegActivationTest, \
    ShiftNegActivationPostAddTest
from tests.keras_tests.feature_networks_tests.feature_networks.softmax_shift_test import SoftmaxShiftTest
from tests.keras_tests.feature_networks_tests.feature_networks.split_concatenate_test import SplitConcatenateTest
from tests.keras_tests.feature_networks_tests.feature_networks.split_conv_bug_test import SplitConvBugTest
from tests.keras_tests.feature_networks_tests.feature_networks.symmetric_threshold_selection_activation_test import \
    SymmetricThresholdSelectionActivationTest
from tests.keras_tests.feature_networks_tests.feature_networks.test_depthwise_conv2d_replacement import \
    DwConv2dReplacementTest
from tests.keras_tests.feature_networks_tests.feature_networks.uniform_range_selection_activation_test import \
    UniformRangeSelectionActivationTest
from tests.keras_tests.feature_networks_tests.feature_networks.weights_mixed_precision_tests import \
    MixedPercisionSearchTest, MixedPercisionDepthwiseTest, \
    MixedPercisionSearchKPI4BitsAvgTest, MixedPercisionSearchKPI2BitsAvgTest, MixedPrecisionActivationDisabled, \
    MixedPercisionSearchLastLayerDistanceTest, MixedPercisionSearchActivationKPINonConfNodesTest, \
    MixedPercisionSearchTotalKPINonConfNodesTest
from tests.keras_tests.infrastructure_tests.base_keras_quantization_wrapper_test import LayerKerasInfrastructureTest

layers = tf.keras.layers


class KerasInfrastructureTest(unittest.TestCase):

    def test_layer_keras_infrastructre(self):
        LayerKerasInfrastructureTest(self).run_test()


if __name__ == '__main__':
    unittest.main()

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
import operator

import unittest
from functools import partial

import numpy as np
import torch
from torch import nn
import model_compression_toolkit as mct
from model_compression_toolkit.core.common.mixed_precision.distance_weighting import MpDistanceWeighting
from model_compression_toolkit.core.common.network_editors import NodeTypeFilter, NodeNameFilter
from model_compression_toolkit.gptq.common.gptq_config import RoundingType
from model_compression_toolkit.target_platform_capabilities import constants as C
from model_compression_toolkit.trainable_infrastructure import TrainingMethod
from tests.pytorch_tests.model_tests.feature_models.add_net_test import AddNetTest
from tests.pytorch_tests.model_tests.feature_models.bn_attributes_quantization_test import BNAttributesQuantization
from tests.pytorch_tests.model_tests.feature_models.compute_max_cut_test import ComputeMaxCutTest
from tests.pytorch_tests.model_tests.feature_models.layer_norm_net_test import LayerNormNetTest
from tests.pytorch_tests.model_tests.feature_models.conv2d_replacement_test import DwConv2dReplacementTest
from tests.pytorch_tests.model_tests.feature_models.manual_bit_selection import ManualBitWidthByLayerTypeTest, \
    ManualBitWidthByLayerNameTest, Manual16BitTest, Manual16BitTestMixedPrecisionTest
from tests.pytorch_tests.model_tests.feature_models.mixed_precision_bops_test import MixedPrecisionBopsBasicTest, \
    MixedPrecisionBopsAllWeightsLayersTest, MixedPrecisionWeightsOnlyBopsTest, MixedPrecisionActivationOnlyBopsTest, \
    MixedPrecisionBopsAndWeightsMemoryUtilizationTest, MixedPrecisionBopsAndActivationMemoryUtilizationTest, MixedPrecisionBopsAndTotalMemoryUtilizationTest, \
    MixedPrecisionBopsWeightsActivationUtilizationTest, MixedPrecisionBopsMultipleOutEdgesTest
from tests.pytorch_tests.model_tests.feature_models.qat_test import QuantizationAwareTrainingTest, \
    QuantizationAwareTrainingMixedPrecisionCfgTest, QuantizationAwareTrainingMixedPrecisionRUCfgTest, \
    QuantizationAwareTrainingQuantizerHolderTest
from tests.pytorch_tests.model_tests.feature_models.relu_replacement_test import SingleLayerReplacementTest, \
    ReluReplacementTest, ReluReplacementWithAddBiasTest
from tests.pytorch_tests.model_tests.feature_models.remove_assert_test import AssertNetTest
from tests.pytorch_tests.model_tests.feature_models.remove_broken_node_test import BrokenNetTest
from tests.pytorch_tests.model_tests.feature_models.concat_threshold_test import ConcatUpdateTest
from tests.pytorch_tests.model_tests.feature_models.add_same_test import AddSameNetTest
from tests.pytorch_tests.model_tests.feature_models.bn_folding_test import BNFoldingNetTest, BNForwardFoldingNetTest
from tests.pytorch_tests.model_tests.feature_models.linear_collapsing_test import TwoConv2DCollapsingTest, \
    ThreeConv2DCollapsingTest, FourConv2DCollapsingTest, SixConv2DCollapsingTest
from tests.pytorch_tests.model_tests.feature_models.residual_collapsing_test import ResidualCollapsingTest1, \
    ResidualCollapsingTest2
from tests.pytorch_tests.model_tests.feature_models.dynamic_size_inputs_test import ReshapeNetTest
from tests.pytorch_tests.model_tests.feature_models.mixed_precision_activation_test import \
    MixedPrecisionActivationSearch8Bit, MixedPrecisionActivationSearch2Bit, MixedPrecisionActivationSearch4Bit, \
    MixedPrecisionActivationSearch4BitFunctional, MixedPrecisionActivationMultipleInputs, \
    MixedPrecisionDistanceFunctions, MixedPrecisionActivationConfigurableWeights
from tests.pytorch_tests.model_tests.feature_models.relu_bound_test import ReLUBoundToPOTNetTest, \
    HardtanhBoundToPOTNetTest
from tests.pytorch_tests.model_tests.feature_models.scalar_tensor_test import ScalarTensorTest
from tests.pytorch_tests.model_tests.feature_models.second_moment_correction_test import ConvSecondMomentNetTest, \
    ConvTSecondMomentNetTest, MultipleInputsConvSecondMomentNetTest, ValueSecondMomentTest
from tests.pytorch_tests.model_tests.feature_models.symmetric_activation_test import SymmetricActivationTest
from tests.pytorch_tests.model_tests.feature_models.test_softmax_shift import SoftmaxLayerNetTest, \
    SoftmaxFunctionNetTest
from tests.pytorch_tests.model_tests.feature_models.permute_substitution_test import PermuteSubstitutionTest
from tests.pytorch_tests.model_tests.feature_models.reshape_substitution_test import ReshapeSubstitutionTest
from tests.pytorch_tests.model_tests.feature_models.constant_conv_substitution_test import ConstantConvSubstitutionTest, \
    ConstantConvReuseSubstitutionTest, ConstantConvTransposeSubstitutionTest
from tests.pytorch_tests.model_tests.feature_models.multi_head_attention_test import MHALayerNetTest, \
    MHALayerNetFeatureTest
from tests.pytorch_tests.model_tests.feature_models.scale_equalization_test import \
    ScaleEqualizationWithZeroPadNetTest, ScaleEqualizationNetTest, \
    ScaleEqualizationReluFuncNetTest, ScaleEqualizationReluFuncWithZeroPadNetTest, \
    ScaleEqualizationConvTransposeWithZeroPadNetTest, ScaleEqualizationReluFuncConvTransposeWithZeroPadNetTest, \
    ScaleEqualizationConvTransposeReluFuncNetTest
from tests.pytorch_tests.model_tests.feature_models.layer_name_test import ReuseNameNetTest
from tests.pytorch_tests.model_tests.feature_models.lut_quantizer_test import LUTWeightsQuantizerTest, \
    LUTActivationQuantizerTest
from tests.pytorch_tests.model_tests.feature_models.mixed_precision_weights_test import MixedPrecisionSearch4Bit, \
    MixedPrecisionActivationDisabledTest, MixedPrecisionSearchLastLayerDistance, MixedPrecisionWithHessianScores, \
    MixedPrecisionSearch8Bit, MixedPrecisionSearchPartWeightsLayers, MixedPrecisionSearch2Bit, \
    MixedPrecisionWeightsConfigurableActivations
from tests.pytorch_tests.model_tests.feature_models.multiple_output_nodes_multiple_tensors_test import \
    MultipleOutputsMultipleTensorsNetTest
from tests.pytorch_tests.model_tests.feature_models.multiple_outputs_node_test import MultipleOutputsNetTest
from tests.pytorch_tests.model_tests.feature_models.output_in_the_middle_test import OutputInTheMiddleNetTest
from tests.pytorch_tests.model_tests.feature_models.parameter_net_test import ParameterNetTest
from tests.pytorch_tests.model_tests.feature_models.reuse_layer_net_test import ReuseLayerNetTest
from tests.pytorch_tests.model_tests.feature_models.shift_negative_activation_test import ShiftNegaviteActivationNetTest
from tests.pytorch_tests.model_tests.feature_models.split_concat_net_test import SplitConcatNetTest
from tests.pytorch_tests.model_tests.feature_models.torch_tensor_attr_net_test import TorchTensorAttrNetTest
from tests.pytorch_tests.model_tests.feature_models.bn_function_test import BNFNetTest
from tests.pytorch_tests.model_tests.feature_models.gptq_test import GPTQAccuracyTest, GPTQWeightsUpdateTest, \
    GPTQLearnRateZeroTest
from tests.pytorch_tests.model_tests.feature_models.uniform_activation_test import \
    UniformActivationTest
from tests.pytorch_tests.model_tests.feature_models.metadata_test import MetadataTest
from tests.pytorch_tests.model_tests.feature_models.tpc_test import TpcTest
from tests.pytorch_tests.model_tests.feature_models.const_representation_test import ConstRepresentationTest, \
    ConstRepresentationMultiInputTest, ConstRepresentationLinearLayerTest, ConstRepresentationGetIndexTest, \
    ConstRepresentationCodeTest
from model_compression_toolkit.target_platform_capabilities.target_platform import QuantizationMethod
from tests.pytorch_tests.model_tests.feature_models.const_quantization_test import ConstQuantizationTest, \
    AdvancedConstQuantizationTest
from tests.pytorch_tests.model_tests.feature_models.remove_identity_test import RemoveIdentityTest
from tests.pytorch_tests.model_tests.feature_models.activation_16bit_test import Activation16BitTest, \
    Activation16BitMixedPrecisionTest


class FeatureModelsTestRunner(unittest.TestCase):

    def test_compute_max_cut(self):
        """
        This test checks the compute max cut of a model and the fused nodes information in the model metadata.
        """
        ComputeMaxCutTest(self).run_test()

    def test_remove_identity(self):
        """
        This test checks that identity layers are removed from the model.
        """
        RemoveIdentityTest(self).run_test()

    def test_single_layer_replacement(self):
        """
        This test checks "EditRule" operation with "ReplaceLayer" action
        Specifically, replacing layer according to its unique name with a custom layer
        """
        SingleLayerReplacementTest(self).run_test()

    def test_relu_replacement(self):
        """
        This test checks "EditRule" operation with "ReplaceLayer" action
        Specifically, replacing layer according to its type with a custom layer
        """
        ReluReplacementTest(self).run_test()

    def test_relu_replacement_with_add_bias(self):
        """
        This test checks "EditRule" operation with "ReplaceLayer" action
        Specifically, replacing layer with a custom layer which is based on the original layer attributes
        """
        ReluReplacementWithAddBiasTest(self).run_test()

    def test_conv2d_replacement(self):
        """
        This test checks "EditRule" operation with "ReplaceLayer" action
        Specifically, updating the weights of Conv2D layer
        """
        DwConv2dReplacementTest(self).run_test()

    def test_add_net(self):
        """
        This test checks the addition and subtraction operations.
        Both with different layers and with constants.
        """
        AddNetTest(self).run_test()

    def test_layer_norm_net(self):
        """
        These tests check the nn.functional.layer_norm operations.
        """
        LayerNormNetTest(self, has_weight=True, has_bias=True).run_test()
        LayerNormNetTest(self, has_weight=True, has_bias=False).run_test()
        LayerNormNetTest(self, has_weight=False, has_bias=True).run_test()
        LayerNormNetTest(self, has_weight=False, has_bias=False).run_test()

    def test_assert_net(self):
        """
        This tests check that the assert operation is being
        removed from the graph during quantization.
        """
        AssertNetTest(self).run_test()

    def test_add_same(self):
        """
        This test checks the special case of addition operation with the same input.
        """
        AddSameNetTest(self, float_reconstruction_error=1e-6).run_test()

    def test_bn_folding(self):
        """
        This test checks the BatchNorm folding feature.
        """
        for functional in [True, False]:
            BNFoldingNetTest(self, nn.Conv2d(3, 2, kernel_size=1), functional, has_weight=False).run_test()
            BNFoldingNetTest(self, nn.Conv2d(3, 3, kernel_size=3, groups=3), functional).run_test()  # DW-Conv test
            BNFoldingNetTest(self, nn.ConvTranspose2d(3, 2, kernel_size=(2, 1)), functional).run_test()
            BNFoldingNetTest(self, nn.Conv2d(3, 2, kernel_size=2), functional, fold_applied=False).run_test()
            BNFoldingNetTest(self, nn.Conv2d(3, 3, kernel_size=(3, 1), groups=3),
                             functional, fold_applied=False).run_test()  # DW-Conv test
            BNFoldingNetTest(self, nn.ConvTranspose2d(3, 2, kernel_size=(1, 3)), functional, fold_applied=False).run_test()
        BNFoldingNetTest(self, nn.ConvTranspose2d(6, 9, kernel_size=(5, 4), groups=3), False).run_test()
        BNFoldingNetTest(self, nn.ConvTranspose2d(3, 3, kernel_size=(4, 2), groups=3), False).run_test()

    def test_bn_forward_folding(self):
        """
        This test checks the BatchNorm forward folding feature.
        """
        BNForwardFoldingNetTest(self, nn.Conv2d(3, 2, 1), is_dw=True).run_test()
        BNForwardFoldingNetTest(self, nn.Conv2d(3, 3, 1, groups=3), is_dw=True).run_test()  # DW-Conv test
        BNForwardFoldingNetTest(self, nn.ConvTranspose2d(3, 2, 1), is_dw=True).run_test()
        BNForwardFoldingNetTest(self, nn.Conv2d(3, 2, 2), fold_applied=False, is_dw=True).run_test()
        BNForwardFoldingNetTest(self, nn.Conv2d(3, 3, (3, 1), groups=3), fold_applied=False,
                                is_dw=True).run_test()  # DW-Conv test
        BNForwardFoldingNetTest(self, nn.ConvTranspose2d(3, 2, (1, 3)), fold_applied=False, is_dw=True).run_test()
        BNForwardFoldingNetTest(self, nn.Conv2d(3, 2, 1), add_bn=True, is_dw=True).run_test()

        BNForwardFoldingNetTest(self, nn.Conv2d(3, 2, 1)).run_test()
        BNForwardFoldingNetTest(self, nn.Conv2d(3, 3, 1, groups=3)).run_test()  # DW-Conv test
        BNForwardFoldingNetTest(self, nn.ConvTranspose2d(3, 2, 1)).run_test()
        BNForwardFoldingNetTest(self, nn.Conv2d(3, 2, 2), fold_applied=False).run_test()
        BNForwardFoldingNetTest(self, nn.Conv2d(3, 3, (3, 1), groups=3), fold_applied=False).run_test()  # DW-Conv test
        BNForwardFoldingNetTest(self, nn.ConvTranspose2d(3, 2, (1, 3)), fold_applied=False).run_test()
        BNForwardFoldingNetTest(self, nn.Conv2d(3, 2, 1), add_bn=True).run_test()

    def test_second_moment_correction(self):
        """
        These are tests for the Second Moment Correction.
        """
        ConvSecondMomentNetTest(self).run_test()
        ConvTSecondMomentNetTest(self).run_test()
        MultipleInputsConvSecondMomentNetTest(self).run_test()
        ValueSecondMomentTest(self).run_test()

    def test_bn_function(self):
        """
        This test checks the batch_norm function.
        """
        BNFNetTest(self).run_test()

    def test_broken_net(self):
        """
        This test checks that the "broken" node (node without output) is being
        removed from the graph during quantization.
        """
        BrokenNetTest(self).run_test()

    def test_linear_collapsing(self):
        """
        This test checks the linear collapsing feature
        """
        TwoConv2DCollapsingTest(self).run_test()
        ThreeConv2DCollapsingTest(self).run_test()
        FourConv2DCollapsingTest(self).run_test()
        SixConv2DCollapsingTest(self).run_test()

    def test_residual_collapsing(self):
        """
        This test checks the residual collapsing feature
        """
        ResidualCollapsingTest1(self).run_test()
        ResidualCollapsingTest2(self).run_test()

    def test_const_quantization(self):
        c = (np.ones((16, 32, 32)) + np.random.random((16, 32, 32))).astype(np.float32)
        for func in [torch.add, torch.sub, torch.mul, torch.div]:
            ConstQuantizationTest(self, func, c).run_test()
            ConstQuantizationTest(self, func, c, input_reverse_order=True).run_test()
            ConstQuantizationTest(self, func, 2.45).run_test()
            ConstQuantizationTest(self, func, 5, input_reverse_order=True).run_test()

        AdvancedConstQuantizationTest(self).run_test()

    def test_const_representation(self):
        for const_dtype in [np.float32, np.int64, np.int32]:
            c = (np.ones((32,)) + np.random.random((32,))).astype(const_dtype)
            c_64 = (np.ones((64,)) + np.random.random((64,))).astype(const_dtype)
            indices = np.random.randint(64, size=32)
            for func in [torch.add, torch.sub, torch.mul, torch.div]:
                ConstRepresentationTest(self, func, c).run_test()
                ConstRepresentationTest(self, func, c, input_reverse_order=True).run_test()
                ConstRepresentationTest(self, func, 2.45).run_test()
                ConstRepresentationTest(self, func, 5, input_reverse_order=True).run_test()
                ConstRepresentationGetIndexTest(self, func, c_64, indices).run_test()

        ConstRepresentationMultiInputTest(self).run_test()

        for enable_weights_quantization in [False, True]:
            c_img = (np.ones((1, 16, 32, 32)) + np.random.random((1, 16, 32, 32))).astype(np.float32)
            ConstRepresentationLinearLayerTest(self, func=nn.Linear(32, 32), const=c_img,
                                               enable_weights_quantization=enable_weights_quantization).run_test()
            ConstRepresentationLinearLayerTest(self, func=nn.Conv2d(16, 16, 1),
                                               const=c_img,
                                               enable_weights_quantization=enable_weights_quantization).run_test()
            ConstRepresentationLinearLayerTest(self, func=nn.ConvTranspose2d(16, 16, 1),
                                               const=c_img, enable_weights_quantization=enable_weights_quantization).run_test()

        ConstRepresentationCodeTest(self).run_test()

    def test_permute_substitution(self):
        """
        This test checks the permute substitution feature
        """
        PermuteSubstitutionTest(self).run_test()
    
    def test_reshape_substitution(self):
        """
        This test checks the reshape substitution feature
        """
        ReshapeSubstitutionTest(self).run_test()
    
    def test_constant_conv_substitution(self):
        """
        This test checks the constant conv substitution feature
        """
        ConstantConvSubstitutionTest(self).run_test()
        ConstantConvReuseSubstitutionTest(self).run_test()
        ConstantConvTransposeSubstitutionTest(self).run_test()

    def test_relu_bound_to_power_of_2(self):
        """
        This test checks the Relu bound to POT feature.
        """
        ReLUBoundToPOTNetTest(self).run_test()

    def test_hardtanh_bound_to_power_of_2(self):
        """
        This test checks the Relu bound to POT feature with Hardtanh layer as Relu.
        """
        HardtanhBoundToPOTNetTest(self).run_test()

    def test_softmax_layer_shift(self):
        """
        This test checks the Softmax shift feature with Softmax as layer.
        """
        SoftmaxLayerNetTest(self).run_test()

    def test_softmax_function_shift(self):
        """
        This test checks the Softmax shift feature with Softmax as function.
        """
        SoftmaxFunctionNetTest(self).run_test()

    def test_scale_equalization(self):
        # This test checks the Channel Scale Equalization feature in Conv2D - Relu - Conv2D with Relu as a layer
        ScaleEqualizationNetTest(self).run_test()
        # This test checks the Channel Scale Equalization feature in Conv2D - Relu - Conv2D with Relu as a layer and with zero padding.
        ScaleEqualizationWithZeroPadNetTest(self).run_test()
        # This test checks the Channel Scale Equalization feature in Conv2D - Relu - Conv2D with Relu as a function
        ScaleEqualizationReluFuncNetTest(self).run_test()
        # This test checks the Channel Scale Equalization feature in Conv2D - Relu - Conv2D with Relu as a function
        # and with zero padding.
        ScaleEqualizationReluFuncWithZeroPadNetTest(self).run_test()
        # This test checks the Channel Scale Equalization feature in ConvTranspose2D - Relu - Conv2D with Relu as a layer
        # and with zero padding.
        ScaleEqualizationConvTransposeWithZeroPadNetTest(self).run_test()
        # This test checks the Channel Scale Equalization feature in ConvTranspose2D - Relu - Conv2D with Relu as a function.
        ScaleEqualizationConvTransposeReluFuncNetTest(self).run_test()
        # This test checks the Channel Scale Equalization feature in Conv2D - Relu - ConvTranspose2D with Relu as a function
        # and with zero padding.
        ScaleEqualizationReluFuncConvTransposeWithZeroPadNetTest(self).run_test()

    def test_scalar_tensor(self):
        """
        This test checks that we support scalar tensors initialized as torch.tensor(x) where x is int
        """
        ScalarTensorTest(self).run_test()

    def test_layer_name(self):
        """
        This test checks that we build a correct graph and correctly reconstruct the model
        given the fact that we reuse nodes and abuse the naming convention of fx (if we resuse
        "conv1" it will be displayed as "conv1_1". So we intentionally add a node named "conv1_1").
        """
        ReuseNameNetTest(self).run_test()

    def test_lut_weights_quantizer(self):
        """
        This test checks multiple features:
        1. That the LUT quantizer quantizes the weights differently than than the Power-of-two quantizer
        2. That the Network Editor works

        In this test we set the weights of 3 conv2d operator to be the same. We set the quantization method
        to "Power-of-two". With the Network Editor we change the quantization method of "conv1" to "LUT quantizer".
        We check that the weights have different values for conv1 and conv2, and that conv2 and conv3 have the same
        values.
        """
        LUTWeightsQuantizerTest(self).run_test()
        LUTWeightsQuantizerTest(self, quant_method=mct.target_platform.QuantizationMethod.LUT_SYM_QUANTIZER).run_test()

    def test_lut_activation_quantizer(self):
        """
        This test checks that activation are quantized correctly using LUT quantizer.
        """
        LUTActivationQuantizerTest(self).run_test()

    def test_multiple_output_nodes_multiple_tensors(self):
        """
        This test checks that we support the connecting the input tensor to several layers
        and taking them as outputs
        """
        MultipleOutputsMultipleTensorsNetTest(self).run_test()

    def test_multiple_outputs_node(self):
        """
        This test checks that we support multiple outputs in the network and multiple
        outputs from a node
        """
        MultipleOutputsNetTest(self).run_test()

    def test_output_in_the_middle(self):
        """
        This test checks:
        That we support taking outputs from the middle of the model.
        """
        OutputInTheMiddleNetTest(self).run_test()

    def test_parameter_net(self):
        """
        This tests check a model with a parameter which is a constant at inference time.
        In addition, the model has an addition layer regular constant tensor
        """
        ParameterNetTest(self).run_test()

    def test_reuse_layer_net(self):
        """
        This test checks:
        The reuse of a layer in a model.
        """
        ReuseLayerNetTest(self).run_test()

    def test_shift_negative_activation_net(self):
        """
        This test checks the shift negative activation feature.
        """
        for activation_layer in [torch.nn.Hardswish, torch.nn.GELU]:
            ShiftNegaviteActivationNetTest(self, activation_layer=activation_layer).run_test(seed=3)

    def test_split_concat_net(self):
        """
        This test checks:
        1. The "split" and "concat" operations.
        2. Nodes with multiple outputs and multiple inputs
        """
        SplitConcatNetTest(self).run_test()

    def test_torch_tensor_attr_net(self):
        """
        This tests checks a model that has calls to torch.Tensor functions,
        such as torch.Tensor.size and torch.Tensor.view.
        """
        TorchTensorAttrNetTest(self).run_test()

    def test_torch_uniform_activation(self):
        """
        This test checks the Uniform activation quantizer.
        """
        UniformActivationTest(self).run_test()

    def test_torch_symmetric_activation(self):
        """
        This test checks the Symmetric activation quantizer.
        """
        SymmetricActivationTest(self).run_test()

    def test_mixed_precision_8bit(self):
        """
        This test checks the Mixed Precision search.
        """
        MixedPrecisionSearch8Bit(self, distance_metric=MpDistanceWeighting.AVG).run_test()
        MixedPrecisionSearch8Bit(self, distance_metric=MpDistanceWeighting.LAST_LAYER).run_test()

    def test_mixed_precision_with_hessian_weights(self):
        """
        This test checks the Mixed Precision search with Hessian-based scores.
        """
        MixedPrecisionWithHessianScores(self, distance_metric=MpDistanceWeighting.AVG).run_test()

    def test_mixed_precision_part_weights_layers(self):
        """
        This test checks the Mixed Precision search.
        """
        MixedPrecisionSearchPartWeightsLayers(self).run_test()

    def test_mixed_precision_2bit(self):
        """
        This test checks the Mixed Precision search.
        """
        MixedPrecisionSearch2Bit(self).run_test()

    def test_reshape_net(self):
        """
        This tests checks the 'reshape_with_static_shapes' substitution.
        We create a model with dynamic input shape attributes to the operators 'reshape' and 'view'.
        We check that the model after conversion replaces the attributes with static-list-of-ints attributes.
        """
        ReshapeNetTest(self).run_test()

    def test_mixed_precision_4bit(self):
        """
        This test checks the Mixed Precision search.
        """
        MixedPrecisionSearch4Bit(self).run_test()

    def test_mixed_precision_with_last_layer_distance(self):
        """
        This test checks the Mixed Precision search with last layer distance function.
        """
        MixedPrecisionSearchLastLayerDistance(self).run_test()

    def test_mixed_precision_activation_disabled(self):
        """
        This test checks the Mixed Precision search.
        """
        MixedPrecisionActivationDisabledTest(self).run_test()

    def test_mixed_precision_activation_8bit(self):
        """
        This test checks the activation Mixed Precision search.
        """
        MixedPrecisionActivationSearch8Bit(self).run_test()

    def test_mixed_precision_activation_2bit(self):
        """
        This test checks the activation Mixed Precision search.
        """
        MixedPrecisionActivationSearch2Bit(self).run_test()

    def test_mixed_precision_activation_4bit(self):
        """
        This test checks the activation Mixed Precision search.
        """
        MixedPrecisionActivationSearch4Bit(self).run_test()

    def test_mixed_precision_activation_only_conf_weights(self):
        """
        This test checks the Mixed Precision for activation only with configurable weights layers.
        """
        MixedPrecisionActivationConfigurableWeights(self).run_test()

    def test_mixed_precision_weights_only_conf_activations(self):
        """
        This test checks the Mixed Precision for weights only with configurable activation layers.
        """
        MixedPrecisionWeightsConfigurableActivations(self).run_test()

    def test_mixed_precision_activation_4bit_functional(self):
        """
        This test checks the activation Mixed Precision search with functional node.
        """
        MixedPrecisionActivationSearch4BitFunctional(self).run_test()

    def test_mixed_precision_multiple_inputs(self):
        """
        This test checks the activation Mixed Precision search with multiple inputs to model.
        """
        MixedPrecisionActivationMultipleInputs(self).run_test()

    def test_mixed_precision_bops_utilization(self):
        """
        This test checks different scenarios for mixed-precision quantization with bit-operations constraint.
        """
        MixedPrecisionBopsBasicTest(self).run_test()
        MixedPrecisionBopsAllWeightsLayersTest(self).run_test()
        MixedPrecisionWeightsOnlyBopsTest(self).run_test()
        MixedPrecisionActivationOnlyBopsTest(self).run_test()
        MixedPrecisionBopsAndWeightsMemoryUtilizationTest(self).run_test()
        MixedPrecisionBopsAndActivationMemoryUtilizationTest(self).run_test()
        MixedPrecisionBopsAndTotalMemoryUtilizationTest(self).run_test()
        MixedPrecisionBopsWeightsActivationUtilizationTest(self).run_test()
        MixedPrecisionBopsMultipleOutEdgesTest(self).run_test()

    def test_mixed_precision_distance_functions(self):
        """
        This test checks the Mixed Precision search with layers that use different distance functions during
        the computation.
        """
        MixedPrecisionDistanceFunctions(self).run_test()

    def test_mha_layer_test(self):
        """
        This test checks the MultiHeadAttentionDecomposition feature.
        """
        num_heads = [3, 7, 5, 11]
        q_seq_len, kv_seq_len = [8, 11, 4, 18], [13, 9, 2, 11]
        qdim, kdim, vdim = [7, 23, 2, 4], [9, None, 7, None], [11, 17, 7, None]
        for iter in range(len(num_heads)):
            MHALayerNetTest(self, num_heads[iter], q_seq_len[iter], qdim[iter] * num_heads[iter],
                            kv_seq_len[iter], kdim[iter], vdim[iter], bias=True).run_test()
            MHALayerNetTest(self, num_heads[iter], q_seq_len[iter], qdim[iter] * num_heads[iter],
                            kv_seq_len[iter], kdim[iter], vdim[iter], bias=False).run_test()
        MHALayerNetFeatureTest(self, num_heads[0], q_seq_len[0], qdim[0] * num_heads[0],
                               kv_seq_len[0], kdim[0], vdim[0], bias=True, add_bias_kv=True).run_test()

    def test_gptq(self):
        """
        This test checks the GPTQ feature.
        """
        GPTQAccuracyTest(self).run_test()
        GPTQAccuracyTest(self, per_channel=False).run_test()
        GPTQAccuracyTest(self, per_channel=True, hessian_weights=False).run_test()
        GPTQAccuracyTest(self, per_channel=True, log_norm_weights=False).run_test()
        GPTQWeightsUpdateTest(self).run_test()
        GPTQLearnRateZeroTest(self).run_test()

        GPTQAccuracyTest(self, rounding_type=RoundingType.SoftQuantizer).run_test()
        GPTQAccuracyTest(self, rounding_type=RoundingType.SoftQuantizer, per_channel=False,
                         params_learning=False).run_test()
        GPTQAccuracyTest(self, rounding_type=RoundingType.SoftQuantizer, per_channel=False,
                         params_learning=True).run_test()
        GPTQAccuracyTest(self, rounding_type=RoundingType.SoftQuantizer,
                         per_channel=True, hessian_weights=True, log_norm_weights=True, scaled_log_norm=True).run_test()
        GPTQWeightsUpdateTest(self, rounding_type=RoundingType.SoftQuantizer).run_test()
        GPTQLearnRateZeroTest(self, rounding_type=RoundingType.SoftQuantizer).run_test()

        GPTQAccuracyTest(self, rounding_type=RoundingType.SoftQuantizer,
                         weights_quant_method=QuantizationMethod.UNIFORM).run_test()
        GPTQAccuracyTest(self, rounding_type=RoundingType.SoftQuantizer,
                         weights_quant_method=QuantizationMethod.UNIFORM, per_channel=False,
                         params_learning=False).run_test()
        GPTQAccuracyTest(self, rounding_type=RoundingType.SoftQuantizer,
                         weights_quant_method=QuantizationMethod.UNIFORM,
                         per_channel=True, hessian_weights=True, log_norm_weights=True, scaled_log_norm=True).run_test()
        GPTQWeightsUpdateTest(self, rounding_type=RoundingType.SoftQuantizer,
                              weights_quant_method=QuantizationMethod.UNIFORM,
                              params_learning=False).run_test()  # TODO: When params learning is True, the uniform quantizer gets a min value  > max value

    def test_qat(self):
        """
        This test checks the QAT feature.
        """
        QuantizationAwareTrainingTest(self).run_test()
        QuantizationAwareTrainingTest(self, finalize=True).run_test()
        _method = mct.target_platform.QuantizationMethod.SYMMETRIC
        QuantizationAwareTrainingTest(self,
                                      weights_quantization_method=_method,
                                      activation_quantization_method=_method
                                      ).run_test()
        QuantizationAwareTrainingTest(self,
                                      weights_quantization_method=_method,
                                      activation_quantization_method=_method,
                                      finalize=True).run_test()
        _method = mct.target_platform.QuantizationMethod.UNIFORM
        QuantizationAwareTrainingTest(self,
                                      weights_quantization_method=_method,
                                      activation_quantization_method=_method
                                      ).run_test()
        QuantizationAwareTrainingTest(self,
                                      weights_quantization_method=_method,
                                      activation_quantization_method=_method,
                                      finalize=True).run_test()
        QuantizationAwareTrainingTest(self,
                                      weights_quantization_method=mct.target_platform.QuantizationMethod.SYMMETRIC,
                                      activation_quantization_method=mct.target_platform.QuantizationMethod.SYMMETRIC,
                                      training_method=TrainingMethod.LSQ,
                                      finalize=True).run_test()
        QuantizationAwareTrainingTest(self,
                                      weights_quantization_method=mct.target_platform.QuantizationMethod.UNIFORM,
                                      activation_quantization_method=mct.target_platform.QuantizationMethod.UNIFORM,
                                      training_method=TrainingMethod.LSQ,
                                      finalize=True).run_test()
        QuantizationAwareTrainingTest(self,
                                      weights_quantization_method=mct.target_platform.QuantizationMethod.POWER_OF_TWO,
                                      activation_quantization_method=mct.target_platform.QuantizationMethod.POWER_OF_TWO,
                                      training_method=TrainingMethod.LSQ,
                                      finalize=True).run_test()
        QuantizationAwareTrainingQuantizerHolderTest(self).run_test()
        QuantizationAwareTrainingMixedPrecisionCfgTest(self).run_test()
        QuantizationAwareTrainingMixedPrecisionRUCfgTest(self).run_test()

    def test_bn_attributes_quantization(self):
        """
        This test checks the quantization of BatchNorm layer attributes.
        """
        BNAttributesQuantization(self, quantize_linear=False).run_test()
        BNAttributesQuantization(self, quantize_linear=True).run_test()
    
    def test_concat_threshold_update(self):
        ConcatUpdateTest(self).run_test()

    def test_metadata(self):
        MetadataTest(self).run_test()

    def test_torch_tpcs(self):
        TpcTest(f'{C.IMX500_TP_MODEL}.v1', self).run_test()
        TpcTest(f'{C.IMX500_TP_MODEL}.v1_lut', self).run_test()
        TpcTest(f'{C.IMX500_TP_MODEL}.v1_pot', self).run_test()
        TpcTest(f'{C.IMX500_TP_MODEL}.v2', self).run_test()
        TpcTest(f'{C.IMX500_TP_MODEL}.v2_lut', self).run_test()
        TpcTest(f'{C.IMX500_TP_MODEL}.v3', self).run_test()
        TpcTest(f'{C.IMX500_TP_MODEL}.v3_lut', self).run_test()
        TpcTest(f'{C.TFLITE_TP_MODEL}.v1', self).run_test()
        TpcTest(f'{C.QNNPACK_TP_MODEL}.v1', self).run_test()

    def test_16bit_activations(self):
        Activation16BitTest(self).run_test()
        Activation16BitMixedPrecisionTest(self).run_test()

    def test_invalid_bit_width_selection(self):
        with self.assertRaises(Exception) as context:
            ManualBitWidthByLayerTypeTest(self, NodeTypeFilter(torch.nn.Conv2d), 7).run_test()
        # Check that the correct exception message was raised
        self.assertEqual(str(context.exception), "Manually selected activation bit-width 7 is invalid for node Conv2d:conv1_bn.")

        with self.assertRaises(Exception) as context:
            ManualBitWidthByLayerTypeTest(self, NodeTypeFilter(operator.add), 3).run_test()
        # Check that the correct exception message was raised
        self.assertEqual(str(context.exception), "Manually selected activation bit-width 3 is invalid for node add:add.")

        with self.assertRaises(Exception) as context:
            ManualBitWidthByLayerNameTest(self, NodeNameFilter('relu'), 3).run_test()
        # Check that the correct exception message was raised
        self.assertEqual(str(context.exception), "Manually selected activation bit-width 3 is invalid for node ReLU:relu.")

    def test_mul_16_bit_manual_selection(self):
        """
        This test checks the execptions in the manual bit-width selection feature.
        """
        # This "mul" can be configured to 16 bit
        Manual16BitTest(self, NodeNameFilter('mul'), 16).run_test()
        Manual16BitTestMixedPrecisionTest(self, NodeNameFilter('mul'), 16).run_test()

        # This "mul" cannot be configured to 16 bit
        with self.assertRaises(Exception) as context:
            Manual16BitTest(self, NodeNameFilter('mul_1'), 16).run_test()
        # Check that the correct exception message was raised
        self.assertEqual(str(context.exception), "Manually selected activation bit-width 16 is invalid for node mul:mul_1.")

        # This "mul" cannot be configured to 16 bit
        with self.assertRaises(Exception) as context:
            Manual16BitTestMixedPrecisionTest(self, NodeNameFilter('mul_1'), 16).run_test()
        # Check that the correct exception message was raised
        self.assertEqual(str(context.exception), "Manually selected activation bit-width 16 is invalid for node mul:mul_1.")

    def test_exceptions__manual_selection(self):
        """
        This test checks the execptions in the manual bit-width selection feature.
        """
        # Node name doesn't exist in graph
        with self.assertRaises(Exception) as context:
            Manual16BitTest(self, NodeNameFilter('mul_3'), 16).run_test()
        # Check that the correct exception message was raised
        self.assertEqual(str(context.exception), "Node Filtering Error: No nodes found in the graph for filter {'node_name': 'mul_3'} to change their bit width to 16.")

        # Invalid inputs to API
        with self.assertRaises(Exception) as context:
            ManualBitWidthByLayerNameTest(self, [NodeNameFilter('relu'), NodeNameFilter('add'), NodeNameFilter('add_1')], [2, 4]).run_test()
        # Check that the correct exception message was raised
        self.assertEqual(str(context.exception),
                         "Configuration Error: The number of provided bit_width values 2 must match the number of filters 3, or a single bit_width value should be provided for all filters.")

    def test_manual_bit_width_selection_by_layer_type(self):
        """
        This test checks the manual bit-width selection feature by layer type filtering.
        """
        ManualBitWidthByLayerTypeTest(self, NodeTypeFilter(torch.nn.Conv2d), 4).run_test()
        ManualBitWidthByLayerTypeTest(self, NodeTypeFilter(torch.nn.Conv2d), 2).run_test()
        ManualBitWidthByLayerTypeTest(self, NodeTypeFilter(torch.nn.Linear), 4).run_test()
        ManualBitWidthByLayerTypeTest(self, NodeTypeFilter(operator.add), 4).run_test()
        ManualBitWidthByLayerTypeTest(self, NodeTypeFilter(operator.add), 2).run_test()
        ManualBitWidthByLayerTypeTest(self, [NodeTypeFilter(torch.nn.Conv2d), NodeTypeFilter(torch.nn.Linear)],
                                      [2, 4]).run_test()
        ManualBitWidthByLayerTypeTest(self, [NodeTypeFilter(torch.nn.Conv2d), NodeTypeFilter(torch.nn.Linear)],
                                      [4, 4]).run_test()
        ManualBitWidthByLayerTypeTest(self, [NodeTypeFilter(torch.nn.Conv2d), NodeTypeFilter(operator.add)],
                                      [2, 4]).run_test()
        ManualBitWidthByLayerTypeTest(self, [NodeTypeFilter(operator.add), NodeTypeFilter(torch.nn.Conv2d)],
                                      [4, 4]).run_test()
        ManualBitWidthByLayerTypeTest(self, [NodeTypeFilter(operator.add), NodeTypeFilter(torch.nn.Linear)],
                                      4).run_test()

    def test_manual_bit_width_selection_by_layer_name(self):
        """
        This test checks the manual bit-width selection feature by layer name filtering.
        """
        ManualBitWidthByLayerNameTest(self, NodeNameFilter('inp'), 4).run_test()
        ManualBitWidthByLayerNameTest(self, NodeNameFilter('conv1_bn'), 4).run_test()
        ManualBitWidthByLayerNameTest(self, NodeNameFilter('fc'), 4).run_test()
        ManualBitWidthByLayerNameTest(self, NodeNameFilter('add'), 4).run_test()
        ManualBitWidthByLayerNameTest(self, NodeNameFilter('add_1'), 4).run_test()
        ManualBitWidthByLayerNameTest(self, NodeNameFilter('conv2_bn'), 4).run_test()
        ManualBitWidthByLayerNameTest(self, NodeNameFilter('relu'), 4).run_test()
        ManualBitWidthByLayerNameTest(self, [NodeNameFilter('add'), NodeNameFilter('conv1_bn')], [2, 4]).run_test()
        ManualBitWidthByLayerNameTest(self, [NodeNameFilter('add'), NodeNameFilter('conv1_bn')], 4).run_test()




if __name__ == '__main__':
    unittest.main()

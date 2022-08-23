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
import unittest

from tests.pytorch_tests.model_tests.feature_models.add_net_test import AddNetTest
from tests.pytorch_tests.model_tests.feature_models.conv2d_replacement_test import DwConv2dReplacementTest
from tests.pytorch_tests.model_tests.feature_models.relu_replacement_test import SingleLayerReplacementTest, \
    ReluReplacementTest, ReluReplacementWithAddBiasTest
from tests.pytorch_tests.model_tests.feature_models.remove_assert_test import AssertNetTest
from tests.pytorch_tests.model_tests.feature_models.remove_broken_node_test import BrokenNetTest
from tests.pytorch_tests.model_tests.feature_models.add_same_test import AddSameNetTest
from tests.pytorch_tests.model_tests.feature_models.bn_folding_test import BNFoldingNetTest
from tests.pytorch_tests.model_tests.feature_models.linear_collapsing_test import TwoConv2DCollapsingTest, \
    ThreeConv2DCollapsingTest, FourConv2DCollapsingTest, SixConv2DCollapsingTest
from tests.pytorch_tests.model_tests.feature_models.residual_collapsing_test import ResidualCollapsingTest1, ResidualCollapsingTest2
from tests.pytorch_tests.model_tests.feature_models.dynamic_size_inputs_test import ReshapeNetTest
from tests.pytorch_tests.model_tests.feature_models.mixed_precision_activation_test import \
    MixedPercisionActivationSearch8Bit, MixedPercisionActivationSearch2Bit, MixedPercisionActivationSearch4Bit, \
    MixedPercisionActivationSearch4BitFunctional
from tests.pytorch_tests.model_tests.feature_models.relu_bound_test import ReLUBoundToPOTNetTest, \
    HardtanhBoundToPOTNetTest
from tests.pytorch_tests.model_tests.feature_models.test_softmax_shift import SoftmaxLayerNetTest, \
    SoftmaxFunctionNetTest
from tests.pytorch_tests.model_tests.feature_models.permute_substitution_test import PermuteSubstitutionTest
from tests.pytorch_tests.model_tests.feature_models.multi_head_attention_test import MHALayerNetTest
from tests.pytorch_tests.model_tests.feature_models.scale_equalization_test import \
    ScaleEqualizationWithZeroPadNetTest, ScaleEqualizationNetTest, \
    ScaleEqualizationReluFuncNetTest, ScaleEqualizationReluFuncWithZeroPadNetTest, \
    ScaleEqualizationConvTransposeWithZeroPadNetTest, ScaleEqualizationReluFuncConvTransposeWithZeroPadNetTest, \
    ScaleEqualizationConvTransposeReluFuncNetTest
from tests.pytorch_tests.model_tests.feature_models.layer_name_test import ReuseNameNetTest
from tests.pytorch_tests.model_tests.feature_models.lut_quantizer_test import LUTWeightsQuantizerTest, \
    LUTActivationQuantizerTest
from tests.pytorch_tests.model_tests.feature_models.mixed_precision_weights_test import MixedPercisionSearch8Bit, \
    MixedPercisionSearch2Bit, MixedPercisionSearch4Bit, MixedPercisionActivationDisabledTest
from tests.pytorch_tests.model_tests.feature_models.multiple_output_nodes_multiple_tensors_test import \
    MultipleOutputsMultipleTensorsNetTest
from tests.pytorch_tests.model_tests.feature_models.multiple_outputs_node_test import MultipleOutputsNetTest
from tests.pytorch_tests.model_tests.feature_models.output_in_the_middle_test import OutputInTheMiddleNetTest
from tests.pytorch_tests.model_tests.feature_models.parameter_net_test import ParameterNetTest
from tests.pytorch_tests.model_tests.feature_models.reuse_layer_net_test import ReuseLayerNetTest
from tests.pytorch_tests.model_tests.feature_models.shift_negative_activation_test import ShiftNegaviteActivationNetTest
from tests.pytorch_tests.model_tests.feature_models.split_concat_net_test import SplitConcatNetTest
from tests.pytorch_tests.model_tests.feature_models.torch_tensor_attr_net_test import TorchTensorAttrNetTest
from tests.pytorch_tests.model_tests.feature_models.layer_fusing_test import LayerFusingTest1, LayerFusingTest2, LayerFusingTest3, LayerFusingTest4
from tests.pytorch_tests.model_tests.feature_models.bn_function_test import BNFNetTest

from tests.pytorch_tests.model_tests.feature_models.gptq_test import GPTQAccuracyTest, GPTQWeightsUpdateTest, GPTQLearnRateZeroTest

class FeatureModelsTestRunner(unittest.TestCase):

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
        This tests check the addition and subtraction operations.
        Both with different layers and with constants.
        """
        AddNetTest(self).run_test()

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
        This test checks the BatchNorm folding feature, plus adding a residual connection.
        """
        BNFoldingNetTest(self).run_test()

    def test_bn_function(self):
        """
        This tests check the batch_norm function and demonstrates the usage of BufferHolder node.
        """
        BNFNetTest(self).run_test()

    def test_broken_net(self):
        """
        This tests checks that the "broken" node (node without output) is being
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

    def test_permute_substitution(self):
        """
        This test checks the permute substitution feature
        """
        PermuteSubstitutionTest(self).run_test()


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
        """
        This test checks the Channel Scale Equalization feature in Conv2D - Relu - Conv2D with Relu as a layer
        """
        ScaleEqualizationNetTest(self).run_test()

    def test_scale_equalization_with_zero_pad(self):
        """
        This test checks the Channel Scale Equalization feature in Conv2D - Relu - Conv2D with Relu as a layer
        and with zero padding.
        """
        ScaleEqualizationWithZeroPadNetTest(self).run_test()

    def test_scale_equalization_with_relu_func(self):
        """
        This test checks the Channel Scale Equalization feature in Conv2D - Relu - Conv2D with Relu as a function
        """
        ScaleEqualizationReluFuncNetTest(self).run_test()

    def test_scale_equalization_with_relu_func_zero_pad(self):
        """
        This test checks the Channel Scale Equalization feature in Conv2D - Relu - Conv2D with Relu as a function
        and with zero padding.
        """
        ScaleEqualizationReluFuncWithZeroPadNetTest(self).run_test()

    def test_scale_equalization_conv_transpose_with_zero_pad(self):
        """
        This test checks the Channel Scale Equalization feature in ConvTranspose2D - Relu - Conv2D with Relu as a layer
        and with zero padding.
        """
        ScaleEqualizationConvTransposeWithZeroPadNetTest(self).run_test()

    def test_scale_equalization_with_relu_func_conv_transpose(self):
        """
        This test checks the Channel Scale Equalization feature in ConvTranspose2D - Relu - Conv2D with Relu as a function.
        """
        ScaleEqualizationConvTransposeReluFuncNetTest(self).run_test()

    def test_scale_equalization_conv_transpose_with_relu_func_zero_pad(self):
        """
        This test checks the Channel Scale Equalization feature in Conv2D - Relu - ConvTranspose2D with Relu as a function
        and with zero padding.
        """
        ScaleEqualizationReluFuncConvTransposeWithZeroPadNetTest(self).run_test()

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

    def test_lut_activation_quantizer(self):
        """
        This test checks that activation are quantized correctly using LUT quantizer
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
        ShiftNegaviteActivationNetTest(self).run_test(seed=3)

    def test_split_concat_net(self):
        """
        This test checks:
        1. The "split" and "concat" operations.
        2. Nodes with multiple outputs and multiple inputs
        """
        SplitConcatNetTest(self).run_test()

    def test_layer_fusing(self):
        """
        This test checks layer fusing: skipping activation quantization for layers in the fusion
        """
        LayerFusingTest1(self).run_test()
        LayerFusingTest2(self).run_test()
        LayerFusingTest3(self).run_test()
        LayerFusingTest4(self).run_test()

    def test_torch_tensor_attr_net(self):
        """
        This tests checks a model that has calls to torch.Tensor functions,
        such as torch.Tensor.size and torch.Tensor.view.
        """
        TorchTensorAttrNetTest(self).run_test()

    def test_mixed_precision_8bit(self):
        """
        This test checks the Mixed Precision search.
        """
        MixedPercisionSearch8Bit(self).run_test()

    def test_mixed_precision_2bit(self):
        """
        This test checks the Mixed Precision search.
        """
        MixedPercisionSearch2Bit(self).run_test()

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
        MixedPercisionSearch4Bit(self).run_test()

    def test_mixed_precision_activation_disabled(self):
        """
        This test checks the Mixed Precision search.
        """
        MixedPercisionActivationDisabledTest(self).run_test()

    def test_mixed_precision_activation_8bit(self):
        """
        This test checks the activation Mixed Precision search.
        """
        MixedPercisionActivationSearch8Bit(self).run_test()

    def test_mixed_precision_activation_2bit(self):
        """
        This test checks the activation Mixed Precision search.
        """
        MixedPercisionActivationSearch2Bit(self).run_test()

    def test_mixed_precision_activation_4bit(self):
        """
        This test checks the activation Mixed Precision search.
        """
        MixedPercisionActivationSearch4Bit(self).run_test()

    def test_mixed_precision_activation_4bit_functional(self):
        """
        This test checks the activation Mixed Precision search with functional node.
        """
        MixedPercisionActivationSearch4BitFunctional(self).run_test()

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


    def test_gptq(self):
        """
        This test checks the GPTQ feature.
        """
        GPTQAccuracyTest(self).run_test()
        GPTQWeightsUpdateTest(self).run_test()
        GPTQLearnRateZeroTest(self).run_test()


if __name__ == '__main__':
    unittest.main()

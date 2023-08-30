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
import model_compression_toolkit as mct
import tensorflow as tf

from mct_quantizers import KerasActivationQuantizationHolder
from model_compression_toolkit.constants import SHIFT_NEGATIVE_NON_LINEAR_NUM_BITS
from model_compression_toolkit.core.common.network_editors import EditRule, node_filters, actions
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import get_keras_tpc_latest
from tests.keras_tests.tpc_keras import get_16bit_tpc
from packaging import version

from tests.keras_tests.utils import get_layers_from_model_by_type

if version.parse(tf.__version__) >= version.parse("2.13"):
    from keras.src.layers.core import TFOpLambda
else:
    from keras.layers.core import TFOpLambda

from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
import numpy as np

keras = tf.keras
layers = keras.layers


class ShiftNegActivationTest(BaseKerasFeatureNetworkTest):
    def __init__(self, unit_test, linear_op_to_test, activation_op_to_test, use_pad_layer=False, input_shape=(8, 8, 3),
                 bypass_op_list=None, param_search=False):
        assert type(linear_op_to_test) in [layers.Conv2D, layers.Dense, layers.DepthwiseConv2D]
        self.linear_op_to_test = linear_op_to_test
        self.activation_op_to_test = activation_op_to_test
        self.use_pad_layer = use_pad_layer
        self.bypass_op_list = bypass_op_list
        self.param_search = param_search
        super().__init__(unit_test, input_shape=input_shape, num_calibration_iter=100, experimental_exporter=True)

    def get_tpc(self):
        return get_16bit_tpc("shift_negative_test")

    def get_quantization_config(self):
        return mct.core.QuantizationConfig(mct.core.QuantizationErrorMethod.MSE, mct.core.QuantizationErrorMethod.MSE,
                                      False, False, weights_per_channel_threshold=True,
                                      shift_negative_activation_correction=True,
                                      shift_negative_ratio=np.inf,
                                      shift_negative_params_search=self.param_search)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = self.activation_op_to_test(inputs)
        if self.bypass_op_list:
            for bypass_op in self.bypass_op_list:
                x = bypass_op(x)
        if self.use_pad_layer:
            x = layers.ZeroPadding2D(((3, 4), (5, 6)))(x)
        outputs = self.linear_op_to_test(x)
        return keras.Model(inputs=inputs, outputs=outputs)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        self.unit_test.assertTrue(float_model.output.shape.as_list() == quantized_model.output.shape.as_list(),
                                  msg=f'Outputs shape mismatch: {float_model.output.shape} != {quantized_model.output.shape}')
        if isinstance(self.activation_op_to_test, tf.keras.layers.PReLU):
            _, w, b = float_model.get_weights()
        else:
            w, b = float_model.get_weights()
        linear_op_index = 3 + (2 if self.use_pad_layer else 1)
        if self.bypass_op_list:
            for bypass_op in self.bypass_op_list:
                # add bypass nodes
                linear_op_index = linear_op_index + 1
                if isinstance(bypass_op, layers.GlobalAveragePooling2D):
                    avg_holder_layer = get_layers_from_model_by_type(quantized_model, KerasActivationQuantizationHolder)[3]
                    self.unit_test.assertTrue(avg_holder_layer.activation_holder_quantizer.get_config()['signed'] == False)

        attr = DEFAULT_KERAS_INFO.get_kernel_op_attributes(type(self.linear_op_to_test))[0]
        linear_layer = get_layers_from_model_by_type(quantized_model, type(self.linear_op_to_test))[0]
        q_w, q_b = linear_layer.get_quantized_weights()[attr].numpy(), linear_layer.layer.bias.numpy()

        # Take the ACTUAL value the activations were shifted by, from the Add layer the substitution needs to insert
        tfoplambda_layers = get_layers_from_model_by_type(quantized_model, TFOpLambda)
        add_layer = [x for x in tfoplambda_layers if x.symbol=='math.add'][0]
        shift_nl_out = add_layer.inbound_nodes[0].call_args[1]
        if isinstance(self.linear_op_to_test, layers.DepthwiseConv2D):
            self.unit_test.assertTrue(np.allclose(b - q_b, shift_nl_out * np.sum(w, axis=(0, 1)).flatten()))
        elif isinstance(self.linear_op_to_test, layers.Conv2D):
            self.unit_test.assertTrue(np.allclose(b - q_b, shift_nl_out * np.sum(w, axis=(0, 1, 2))))
        elif isinstance(self.linear_op_to_test, layers.Dense):
            self.unit_test.assertTrue(np.allclose(b - q_b, shift_nl_out * np.sum(w, axis=0)))
        else:
            raise NotImplementedError


class ShiftNegActivationPostAddTest(ShiftNegActivationTest):
    """
    This test is meant to verify that when using shift negative correction, the post_add layer that is added to the
    model after the non-linear layer that its activations' are bing shifted,
    has the correct quantization number of bits.
    It also verifies that the non-linear layer's is quantized with the number of bits that is
    representative of a float model (since we only want to quantize the post_add layer's activations)
    """
    def __init__(self, unit_test, linear_op_to_test, activation_op_to_test, post_add_nbits=7):
        super().__init__(unit_test, linear_op_to_test, activation_op_to_test)

        self.post_add_nbits = post_add_nbits

    def get_tpc(self):
        return get_keras_tpc_latest()

    def get_debug_config(self):
        return mct.core.DebugConfig(network_editor=[EditRule(filter=node_filters.NodeNameScopeFilter('activation'),
                                                        action=actions.ChangeCandidatesActivationQuantConfigAttr(
                                                            activation_n_bits=self.post_add_nbits))])

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        self.unit_test.assertTrue(float_model.output.shape.as_list() == quantized_model.output.shape.as_list(),
                                  msg=f'Outputs shape mismatch: {float_model.output.shape} != {quantized_model.output.shape}')

        non_linear_layer_fake_quant = get_layers_from_model_by_type(quantized_model, KerasActivationQuantizationHolder)[1]
        non_linear_nbits = non_linear_layer_fake_quant.activation_holder_quantizer.get_config()['num_bits']
        self.unit_test.assertTrue(non_linear_nbits == SHIFT_NEGATIVE_NON_LINEAR_NUM_BITS,
                                  f"The non-linear node's activation_n_bits after applying snc should be "
                                  f"{SHIFT_NEGATIVE_NON_LINEAR_NUM_BITS}, but activation_n_bits is {non_linear_nbits}")

        post_add_layer_fake_quant = get_layers_from_model_by_type(quantized_model, KerasActivationQuantizationHolder)[2]
        post_add_nbits = post_add_layer_fake_quant.activation_holder_quantizer.get_config()['num_bits']
        self.unit_test.assertTrue(post_add_nbits == self.post_add_nbits,
                                  f"The post_add layer that's added after the non-linear node "
                                  f"should be quantized with {self.post_add_nbits}, "
                                  f"but activation_n_bits is {post_add_nbits}")

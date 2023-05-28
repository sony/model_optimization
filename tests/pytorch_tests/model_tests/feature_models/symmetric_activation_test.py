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

import numpy as np
import torch

import model_compression_toolkit as mct
from model_compression_toolkit.constants import THRESHOLD
from mct_quantizers import QuantizationMethod
from model_compression_toolkit.core.common.user_info import UserInformation
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor
from model_compression_toolkit.target_platform_capabilities.tpc_models.default_tpc.latest import generate_pytorch_tpc
from tests.common_tests.helpers.generate_test_tp_model import generate_test_tp_model
from tests.pytorch_tests.model_tests.base_pytorch_test import BasePytorchTest

"""
This test checks the Symmetric activation quantizer.
"""


def weight_change(layer: torch.nn.Module):
    w_shape = layer.weight.shape
    b_shape = layer.bias.shape
    delattr(layer, 'weight')
    delattr(layer, 'bias')
    layer.register_buffer('weight', torch.ones(w_shape))
    layer.register_buffer('bias', torch.zeros(b_shape))
    return layer


class SymmetricActivationTest(BasePytorchTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)
        self.const_input = 3

    def get_tpc(self):
        tp = generate_test_tp_model({
            'activation_quantization_method': QuantizationMethod.SYMMETRIC,
            "enable_weights_quantization": False,
            'activation_n_bits': 8})
        return {'act_8bit': generate_pytorch_tpc(name="symmetric_layer_test", tp_model=tp)}

    def get_core_configs(self):
        qc = mct.core.QuantizationConfig(mct.core.QuantizationErrorMethod.NOCLIPPING,
                                    mct.core.QuantizationErrorMethod.NOCLIPPING,
                                    shift_negative_activation_correction=True,
                                    shift_negative_ratio=np.inf)
        return {'act_8bit': mct.core.CoreConfig(quantization_config=qc)}

    def create_feature_network(self, input_shape):
        return SymmetricActivationNet(input_shape)

    def create_inputs_shape(self):
        return [[self.val_batch_size, 1, 224, 224]]

    def generate_inputs(self, input_shapes):
        # generate output between - self.const and 2 * self.const
        return to_torch_tensor(
            [self.const_input * (torch.randint(0, 4, in_shape, dtype=torch.float32) - 1.0) for in_shape in
             input_shapes])

    def compare(self, quantized_models, float_model, input_x=None, quantization_info: UserInformation = None):
        for model_name, quantized_model in quantized_models.items():

            # const = 3
            # The range of the input values is from -const to 2*const, the threshold needs to be 2*const.
            # Because the symmetric quantization the activations values needs to be
            # between -2*const to (2*const - delta).
            # We'll pass an input with lower values than the original input, and see that the activations values will
            # be quantized to -2*const.
            output = quantized_model(10*input_x[0]).cpu().detach().numpy()

            # check the threshold value is not POT
            output_layer_threshold = quantized_model.node_sort[
                -1].final_activation_quantization_cfg.activation_quantization_params[THRESHOLD]
            output_layer_threshold_log = np.log2(output_layer_threshold)
            self.unit_test.assertFalse(
                np.array_equal(output_layer_threshold_log, output_layer_threshold_log.astype(int)),
                msg=f"Output threshold {output_layer_threshold} is a power of 2")

            # check the output is bounded by the symmetric threshold
            # In symmetric quantization max value is activation_threshold * (1 - 1 / (2 ** (activation_n_bits - 1))
            max_output = output.max()
            activation_n_bits = quantized_model.node_sort[-1].final_activation_quantization_cfg.activation_n_bits
            delta = output_layer_threshold / 2 ** (activation_n_bits - 1)
            self.unit_test.assertTrue(max_output == output_layer_threshold - delta)

            # check the activations is bounded by the threshold
            min_output = output.min()
            self.unit_test.assertTrue(np.abs(min_output) == output_layer_threshold)


class SymmetricActivationNet(torch.nn.Module):
    def __init__(self, input_shape):
        super(SymmetricActivationNet, self).__init__()
        _, in_channels, _, _ = input_shape[0]
        self.layer1 = torch.nn.Conv2d(in_channels, 1, bias=True, kernel_size=(1, 1))
        self.layer1 = weight_change(self.layer1)

    def forward(self, inp):
        x = self.layer1(inp)
        return x

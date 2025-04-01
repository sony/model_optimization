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
import math

import numpy as np
import pytest
import torch
from mct_quantizers import QuantizationMethod
from torch import nn

from model_compression_toolkit.core import QuantizationErrorMethod, QuantizationConfig, CoreConfig
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor
from model_compression_toolkit.ptq import pytorch_post_training_quantization
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import OpQuantizationConfig, \
    AttributeQuantizationConfig, Signedness
from model_compression_toolkit.target_platform_capabilities.constants import KERNEL_ATTR, BIAS_ATTR
from tests.common_tests.helpers.tpcs_for_tests.v4.tpc import generate_tpc

INPUT_SHAPE = (1, 8, 12, 8)
torch.manual_seed(42)


def build_model(input_shape, out_chan, kernel, const):
    """
    Build a simple CNN model with a convolution layer and ReLU activation.

    The convolution weights are initialized with a constant normalized sequential tensor.
    """
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(in_channels=input_shape[1], out_channels=out_chan, kernel_size=kernel)
            self.relu = nn.ReLU()

            # Initialize convolution weights with sequential numbers from 1 to total number of weights
            total_weights = self.conv.weight.numel()
            # Create a tensor with values from 1 to total_weights and reshape it to the shape of conv weights
            weights_seq = torch.arange(1, total_weights + 1, dtype=self.conv.weight.dtype).reshape(
                self.conv.weight.shape)
            weights_seq = weights_seq/weights_seq.sum()
            with torch.no_grad():
                self.conv.weight.copy_(weights_seq)

        def forward(self, x):
            x = self.conv(x)
            x = self.relu(x)
            x = torch.add(x, to_torch_tensor(const))
            return x

    return Model()


@pytest.fixture
def rep_data_gen():
    """
    Fixture to create a representative dataset generator for post-training quantization.

    Generates a small dataset based on the defined INPUT_SHAPE.
    """
    np.random.seed(42)

    def representative_dataset():
        for _ in range(2):
            yield [np.random.randn(*INPUT_SHAPE)]

    return representative_dataset


def get_tpc():
    """
    Create a target platform capabilities (TPC) configuration with no weight quantization.

    Returns a TPC object for quantization tests.
    """
    att_cfg_noquant = AttributeQuantizationConfig()

    op_cfg = OpQuantizationConfig(default_weight_attr_config=att_cfg_noquant,
                                  attr_weights_configs_mapping={KERNEL_ATTR: att_cfg_noquant,
                                                                BIAS_ATTR: att_cfg_noquant},
                                  activation_quantization_method=QuantizationMethod.UNIFORM,
                                  activation_n_bits=2,
                                  supported_input_activation_n_bits=2,
                                  enable_activation_quantization=True,
                                  quantization_preserving=True,
                                  fixed_scale=None,
                                  fixed_zero_point=None,
                                  simd_size=32,
                                  signedness=Signedness.AUTO)

    tpc = generate_tpc(default_config=op_cfg, base_config=op_cfg, mixed_precision_cfg_list=[op_cfg], name="test_tpc")

    return tpc


def get_core_config(quant_error_method):
    """
    Create a core configuration with a specified quantization error method.

    Parameters:
        quant_error_method: QuantizationErrorMethod to be used in the configuration.

    Returns:
        CoreConfig instance configured with the specified quantization error method.
    """
    quantization_config = QuantizationConfig(activation_error_method=quant_error_method)
    return CoreConfig(quantization_config=quantization_config)


class TestPTQWithQuantizationErrorMethods:
    def test_ptq_quantization_error_methods(self, rep_data_gen):
        """
        Verify that post-training quantization produces different quantization parameters based
        on the chosen quantization error method.

        For each quantization error method (MSE, HMSE, MAE, NOCLIPPING), this test builds a model,
        applies post-training quantization, and records the max_range of the ReLU activation quantizer.
        It then asserts that each method produces a distinct quantization parameter.
        """
        model = build_model(input_shape=INPUT_SHAPE, out_chan=16, kernel=1, const=np.array([5]))
        tpc = get_tpc()
        quant_max_range={}
        quant_error_method_types = [QuantizationErrorMethod.MSE, QuantizationErrorMethod.HMSE, QuantizationErrorMethod.MAE, QuantizationErrorMethod.NOCLIPPING]
        for quant_error_method in quant_error_method_types:
            q_model, quantization_info = pytorch_post_training_quantization(in_module=model,
                                                                            core_config=get_core_config(quant_error_method),
                                                                            representative_data_gen=rep_data_gen,
                                                                            target_platform_capabilities=tpc)
            quant_max_range[quant_error_method] = q_model.relu_activation_holder_quantizer.activation_holder_quantizer.max_range

        # Ensure that each quantization error method yields a unique quantization parameter.
        assert len(np.unique(list(quant_max_range.values()))) == len(quant_max_range)
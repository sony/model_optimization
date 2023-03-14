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
import random

import numpy as np
import torch
import torch.nn as nn

import model_compression_toolkit as mct
from model_compression_toolkit.core.common.target_platform import TargetPlatformCapabilities
from model_compression_toolkit.core.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO
from tests.pytorch_tests.model_tests.base_pytorch_test import BasePytorchTest

"""
This test checks the MultiHeadAttentionDecomposition feature.
"""


class MHABaseTest(BasePytorchTest):
    def __init__(self, unit_test, num_heads, q_seq_len, embed_dim, kv_seq_len, kdim=None, vdim=None, bias=True,
                 add_bias_kv=False, add_zero_attn=False, batch_first=True, float_reconstruction_error=1e-6):
        super().__init__(unit_test, float_reconstruction_error)

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.q_dim = int(embed_dim / num_heads)
        self.query_input_shape = (q_seq_len, self.embed_dim)
        self.kdim = kdim
        self.vdim = vdim
        self.kdim_for_input = kdim if kdim is not None else embed_dim
        self.vdim_for_input = vdim if vdim is not None else embed_dim
        self.kv_seq_len = kv_seq_len
        self.key_input_shape = (self.kv_seq_len, self.kdim_for_input)
        self.value_input_shape = (self.kv_seq_len, self.vdim_for_input)
        self.bias = bias
        self.add_bias_kv = add_bias_kv
        self.add_zero_attn = add_zero_attn
        self.batch_first = batch_first

    def create_inputs_shape(self):
        return [[self.val_batch_size] + list(self.query_input_shape),
                [self.val_batch_size] + list(self.key_input_shape),
                [self.val_batch_size] + list(self.value_input_shape)]


class MHANet(nn.Module):
    # This network based on single MHA layer
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True,
                 add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None,
                 batch_first=True, device=None, dtype=None):
        super(MHANet, self).__init__()
        self.batch_first = batch_first
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, bias=bias,
                                         add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn, kdim=kdim, vdim=vdim,
                                         batch_first=batch_first, device=device, dtype=dtype)

    def forward(self, q, k, v):
        x = self.mha(q, k, v)
        return x


class MHALayerNetTest(MHABaseTest):
    """
    This test checks the MultiHeadAttention as a single layer.
    """

    def create_feature_network(self, input_shape):
        return MHANet(embed_dim=self.embed_dim, num_heads=self.num_heads, kdim=self.kdim,
                      vdim=self.vdim, bias=self.bias, add_bias_kv=self.add_bias_kv,
                      add_zero_attn=self.add_zero_attn, batch_first=self.batch_first)


class MHALayerNetFeatureTest(MHALayerNetTest):
    """
    This test checks the MultiHeadAttention as a single layer with add_bias_kv feature.
    """

    def run_test(self, seed=0):
        np.random.seed(seed)
        random.seed(a=seed)
        torch.random.manual_seed(seed)
        input_shapes = self.create_inputs_shape()
        x = self.generate_inputs(input_shapes)

        def representative_data_gen():
            for _ in range(self.num_calibration_iter):
                yield x

        model_float = self.create_feature_network(input_shapes)
        tpc_dict = self.get_tpc()
        core_configs_dict = self.get_core_configs()
        assert isinstance(tpc_dict, dict), "Pytorch tests get_tpc should return a dictionary " \
                                           "mapping the test model name to a TPC object."
        for model_name, tpc in tpc_dict.items():
            assert isinstance(tpc, TargetPlatformCapabilities)
            assert model_name in core_configs_dict
            core_config = core_configs_dict[model_name]

            with self.unit_test.assertRaises(Exception) as e:
                ptq_model, quant_info = mct.pytorch_post_training_quantization_experimental(model_float,
                                                                                            representative_data_gen,
                                                                                            core_config=core_config,
                                                                                            target_platform_capabilities=tpc)
            self.unit_test.assertEqual('Add BIAS_KV feature is Not Implemented', str(e.exception))

# Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
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

from tests.pytorch_tests.model_tests.base_pytorch_test import BasePytorchTest
from torch import nn
from packaging import version
import torch

class ScaledDotProductAttentionNet(nn.Module):
    def __init__(self, dropout_p=0.0, scale=None, attn_mask=None, is_causal=False):
        super().__init__()
        self.dropout_p = dropout_p
        self.scale = scale
        self.attn_mask = attn_mask
        self.is_causal = is_causal

    def forward(self, q, k, v):
        x = nn.functional.scaled_dot_product_attention(q, k, v,
                                                       attn_mask=self.attn_mask,
                                                       dropout_p=self.dropout_p,
                                                       is_causal=self.is_causal,
                                                       # scale=self.scale
                                                       )
        return x


class ScaledDotProductAttentionTest(BasePytorchTest):
    """
    This test checks the scaled_dot_product_attention (SDPA) substitution using a single SDPA layer.
    """

    def __init__(self, unit_test, batch_size: int, q_and_k_embd_size: int, v_embd_size: int, source_seq_len: int,
                 target_seq_len: int, dropout_p: float = 0.0, scale: float = None, attn_mask: float = None,
                 is_causal: bool = False):

        super().__init__(unit_test)
        self.batch_size = batch_size
        self.q_and_k_embd_size = q_and_k_embd_size
        self.v_embd_size = v_embd_size
        self.source_seq_len = source_seq_len
        self.target_seq_len = target_seq_len
        self.use_is_close_validation = True  # because SDPA contains sqrt operation which leads to sightly different output values compared to original torch model
        self.dropout_p = dropout_p
        self.scale = scale
        self.attn_mask = attn_mask
        self.is_causal = is_causal

    def create_feature_network(self, input_shape):

        if version.parse(torch.__version__) >= version.parse("2.1"):
            return ScaledDotProductAttentionNet(dropout_p=self.dropout_p,
                                                attn_mask=self.attn_mask,
                                                is_causal=self.is_causal,
                                                scale=self.scale)

        else:  # older torch versions don't have scale argument
            return ScaledDotProductAttentionNet(dropout_p=self.dropout_p,
                                                attn_mask=self.attn_mask,
                                                is_causal=self.is_causal)

    def create_inputs_shape(self):
        q_shape = [self.batch_size, self.target_seq_len, self.q_and_k_embd_size]
        k_shape = [self.batch_size, self.source_seq_len, self.q_and_k_embd_size]
        v_shape = [self.batch_size, self.source_seq_len, self.v_embd_size]
        return [q_shape, k_shape, v_shape]

    def _test_substitution_structure_output(self, post_substitution_nodes):
        """
        :param post_substitution_nodes: The graph nodes after the SDPA substitution
        raise Exception if case the post_substitution_nodes doesn't match the expected_nodes_counter
        """
        expected_nodes_counter = {
            'DummyPlaceHolder': 3,
            "transpose": 1,
            "matmul": 2,
            "mul": 1,  # scale operator
            "Softmax": 1,
            "Dropout": 1,
            "add": 0 if self.attn_mask is None else 1  # mask operator
        }

        for node in post_substitution_nodes:
            operator_name = node.layer_class.__name__
            if not (operator_name in expected_nodes_counter):
                raise Exception(f"Post substitution graph contains unexpected node: {operator_name}")
            expected_nodes_counter[operator_name] -= 1

        counter_results = set(expected_nodes_counter.values())
        if not (len(counter_results) == 1 and 0 in counter_results):  # validate that all values are zeros
            raise Exception(f"Post substitution graph contains unexpected nodes: {[k for k, v in expected_nodes_counter.items() if v != 0]}")

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        super().compare(quantized_models, float_model, input_x, quantization_info)
        post_substitution_nodes = quantized_models['no_quantization'].node_sort
        self._test_substitution_structure_output(post_substitution_nodes)

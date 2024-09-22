from tests.pytorch_tests.model_tests.base_pytorch_test import BasePytorchTest
from torch import nn


class ScaledDotProductAttentionNet(nn.Module):
    def __init__(self, attn_mask=None, dropout_p=0.0):
        super().__init__()
        self.dropout_p = dropout_p
        self.attn_mask = attn_mask

    def forward(self, q, k, v):
        x = nn.functional.scaled_dot_product_attention(q, k, v,
                                                       dropout_p=self.dropout_p,
                                                       attn_mask=self.attn_mask
                                                       )
        return x


class ScaledDotProductAttentionTest(BasePytorchTest):
    """
    This test checks the MultiHeadAttention as a single layer with add_bias_kv feature.
    """
    def __init__(self, unit_test, attn_mask=None, dropout_p=0.0):
        super().__init__(unit_test)
        self.dropout_p = dropout_p
        self.attn_mask = attn_mask

    def create_feature_network(self, input_shape):
        return ScaledDotProductAttentionNet(self.attn_mask, self.dropout_p)

    def create_inputs_shape(self):
        batch_size, seq_len, embd_size = 1, 8, 21
        q = [batch_size, 8, 21]  # batch_size, seq_len. embd_size
        k = [batch_size, 8, 21]  # batch_size, seq_len. embd_size
        v = [batch_size, 8, 21]  # batch_size, seq_len. embd_size
        # k = [batch_size, 13, 9]
        # v = [batch_size, 13, 11]
        # returns query, key and value tensor shapes
        return [q, k, v]

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        super()
        print("a")
        q_nodes = quantized_models['all_4bit'].node_sort
        # assert "activation_post_add" in [n.name for n in q_nodes], "Add operator haven't been added after activation operator"
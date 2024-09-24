from tests.pytorch_tests.model_tests.base_pytorch_test import BasePytorchTest
from torch import nn


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
                                                       scale=self.scale
                                                       )
        return x


class ScaledDotProductAttentionTest(BasePytorchTest):
    """
    This test checks the MultiHeadAttention as a single layer with add_bias_kv feature.
    """
    def __init__(self, unit_test, dropout_p=0.0, scale=None, attn_mask=None, is_causal=False):
        super().__init__(unit_test)
        self.use_fuzzy_validation = True  # because SDPA contains sqrt operation which leads to sightly different output values compared to original torch model
        self.dropout_p = dropout_p
        self.scale = scale
        self.attn_mask = attn_mask
        self.is_causal = is_causal

    def create_feature_network(self, input_shape):
        return ScaledDotProductAttentionNet(dropout_p=self.dropout_p,
                                            attn_mask=self.attn_mask,
                                            is_causal=self.is_causal,
                                            scale=self.scale)

    def create_inputs_shape(self):
        batch_size, q_and_k_embd_size, v_embd_size, source_seq_len, target_seq_len = 3, 8, 19, 21, 13
        q_shape = [batch_size, target_seq_len, q_and_k_embd_size]
        k_shape = [batch_size, source_seq_len, q_and_k_embd_size]
        v_shape = [batch_size, source_seq_len, v_embd_size]
        return [q_shape, k_shape, v_shape]

    def _test_substitution_structure_output(self, post_substitution_nodes):
        """
        :param orig_graph: The original float model graph before substitution
        :param new_graph: The post substitutions graph
        :return: True if the new graph after scaled_dot_product_attention substitution is in the correct structure.
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

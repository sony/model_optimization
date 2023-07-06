# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
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
from typing import List, Callable, Any

import numpy as np

from model_compression_toolkit.core.common.quantization.candidate_node_quantization_config import \
    CandidateNodeQuantizationConfig


def verify_candidates_descending_order(node_q_cfg: List[CandidateNodeQuantizationConfig]):
    """
    Make sure the candidates configurations arrives in descending order.

    Args:
        node_q_cfg: Quantization configuration candidates of the node that generated the layer that will
                    use this quantizer.

    Returns:

    """
    curmax = (np.inf, np.inf)
    n_candidate_bits = [(x.weights_quantization_cfg.weights_n_bits, x.activation_quantization_cfg.activation_n_bits)
                        for x in node_q_cfg]
    for candidate_bits in n_candidate_bits:
        assert candidate_bits < curmax, f"Node's quantization configuration candidates should arrive in " \
                                        f"descending order of (weights_nbits, activation_nbits)."
        curmax = candidate_bits


def init_quantized_weights(node_q_cfg: List[CandidateNodeQuantizationConfig],
                           float_weights: Any,
                           fw_tensor_convert_func: Callable) -> List:
    """
    Initilizes quantized weights tensors according to the given quantization configuration candidates.

    Args:
        node_q_cfg: Quantization configuration candidates of the node that generated the layer that will
                    use this quantizer.
        float_weights: A tensor of the layer's weights.
        fw_tensor_convert_func: A function that converts a tensor to a framework specific tensor type.

    Returns: A list with the quantized weights for each candidate.

    """

    quantized_weights = []
    for qc in node_q_cfg:
        qc_weights = qc.weights_quantization_cfg
        q_weight = qc_weights.weights_quantization_fn(float_weights,
                                                      qc_weights.weights_n_bits,
                                                      True,
                                                      qc_weights.weights_quantization_params,
                                                      qc_weights.weights_per_channel_threshold,
                                                      qc_weights.weights_channels_axis)

        quantized_weights.append(fw_tensor_convert_func(q_weight))

    return quantized_weights


def init_activation_quantizers(node_q_cfg: List[CandidateNodeQuantizationConfig]) -> List:
    """
    Builds a list of quantizers for each of the bitwidth candidates for activation quantization,
    to be stored and used during MP search.

    Args:
        node_q_cfg: Quantization configuration candidates of the node that generated the layer that will
                    use this quantizer.

    Returns: a list of activation quantizers - for each bitwidth and layer's attribute to be quantized.
    """

    activation_quantizers = []
    for index, qc in enumerate(node_q_cfg):
        q_activation = node_q_cfg[index].activation_quantization_cfg
        activation_quantizers.append(q_activation.quantize_node_output)

    return activation_quantizers

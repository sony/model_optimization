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


from tensorflow.keras.layers import DepthwiseConv2D, Conv2D, Conv2DTranspose, Dense
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit.default_8bit_quantize_registry import \
    QuantizeConfig

from model_compression_toolkit.core import common
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.core.keras.quantizer.mixed_precision.selective_quantize_config import \
    SelectiveQuantizeConfig



def quantization_config_builder_mixed_precision(n: common.BaseNode) -> QuantizeConfig:
    """
    Build a QuantizeConfig for layers that should be wrapped in a QuantizeWrapper to
    be part of a mixed-precision model.

    Args:
        n: Node to build its QuantizeConfig.

    Returns:
        QuantizeConfig to wrap the layer so it can work in MP Keras models.
    """

    assert n.candidates_quantization_cfg is not None
    node_q_cfg_candidates = n.candidates_quantization_cfg
    # sort by decending bit width so using indices would be easier
    node_q_cfg_candidates.sort(key=lambda x: (x.weights_quantization_cfg.weights_n_bits,
                                              x.activation_quantization_cfg.activation_n_bits), reverse=True)

    float_weights = [n.get_weights_by_keys(attr) for attr in DEFAULT_KERAS_INFO.get_kernel_op_attributes(n.type)]

    max_cfg_candidates = n.find_max_candidates_indices()
    assert len(max_cfg_candidates) == 1, \
        f"A maximal config candidate must be defined, but some node have multiple potential maximal candidates"
    max_candidate_idx = max_cfg_candidates[0]

    # Create a SelectiveQuantizeConfig that holds the float and quantized weights (every weight is
    # quantized using all possible bitwidhts in the node's candidates weights quantization configurations).
    return SelectiveQuantizeConfig(node_q_cfg=node_q_cfg_candidates,
                                   float_weights=float_weights,
                                   weight_attrs=DEFAULT_KERAS_INFO.get_kernel_op_attributes(n.type),
                                   max_candidate_idx=max_candidate_idx)

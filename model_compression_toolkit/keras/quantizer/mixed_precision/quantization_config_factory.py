# Copyright 2021 Sony Semiconductors Israel, Inc. All rights reserved.
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

from model_compression_toolkit import common
from model_compression_toolkit.common.framework_info import FrameworkInfo
from model_compression_toolkit.keras.quantizer.mixed_precision.selective_weights_quantize_config import \
    SelectiveWeightsQuantizeConfig



def quantization_config_builder_mixed_precision(n: common.BaseNode,
                                                fw_info: FrameworkInfo) -> QuantizeConfig:
    """
    Build a QuantizeConfig for layers that should be wrapped in a QuantizeWrapper to
    be part of a mixed-precision model.

    Args:
        n: Node to build its QuantizeConfig.
        fw_info: Framework information (e.g., mapping from layers to their attributes to quantize).

    Returns:
        QuantizeConfig to wrap the layer so it can work in MP Keras models.
    """

    assert n.candidates_weights_quantization_cfg is not None
    node_weights_q_cfg = n.candidates_weights_quantization_cfg
    # sort by decending bit width so using indices would be easier
    node_weights_q_cfg.sort(key=lambda x: x.weights_n_bits, reverse=True)

    float_weights = [n.get_weights_by_keys(attr) for attr in fw_info.get_kernel_op_attributes(n.layer_class)]

    # Create a SelectiveWeightsQuantizeConfig that holds the float and quantized weights (every weight is
    # quantized using all possible bitwidhts in the node's candidates weights quantization configurations).
    return SelectiveWeightsQuantizeConfig(fw_info.get_kernel_op_attributes(n.layer_class),
                                          float_weights=float_weights,
                                          node_weights_q_cfg=node_weights_q_cfg)

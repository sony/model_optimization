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


from tensorflow.keras.layers import DepthwiseConv2D, Conv2D, Conv2DTranspose
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit.default_8bit_quantize_configs import \
    NoOpQuantizeConfig
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit.default_8bit_quantize_registry import \
    QuantizeConfig

from model_compression_toolkit import common
from model_compression_toolkit import keras
from model_compression_toolkit.common.constants import THRESHOLD
from model_compression_toolkit.keras.constants import DEPTHWISE_KERNEL, KERNEL

KERNEL_NAME = {Conv2DTranspose: KERNEL,
               DepthwiseConv2D: DEPTHWISE_KERNEL,
               Conv2D: KERNEL}


def quantization_config_builder_kd(n: common.Node) -> QuantizeConfig:
    """
    Build a QuantizeConfig for a node according to its quantization configuration and
    a global NoOpQuantizeConfig object.

    Args:
        n: Node to build its QuantizeConfig.

    Returns:
        A QuantizeConfig object with the appropriate quantizers (according to the node's
        quantization configuration).
    """

    node_activation_q_cfg = n.activation_quantization_cfg  # get configs from node
    node_weights_q_cfg = n.weights_quantization_cfg

    # If the node's output should be quantized, use a QuantizeConfig with an activation quantizer only.
    if node_activation_q_cfg is not None:
        if node_activation_q_cfg.has_activation_quantization_params():
            if node_activation_q_cfg.enable_activation_quantization:
                return keras.quantizer.ActivationQuantizeConfigKD(node_activation_q_cfg.activation_quantization_params,
                                                                  node_activation_q_cfg.activation_is_signed,
                                                                  num_bits=node_activation_q_cfg.activation_n_bits)
            else:
                return NoOpQuantizeConfig()

    # If the node's output and weights should be quantized, it is not supported.
    # If according to the quantization configuration no quantization is enabled - use a
    # QuantizeConfig which does not quantize.
    elif n.activation_weights_quantization():
            if node_activation_q_cfg.enable_activation_quantization and node_weights_q_cfg.enable_weights_quantization:
                raise NotImplemented
            else:
                return NoOpQuantizeConfig()

    # If the node's weights should be quantized, use a QuantizeConfig with a weights quantizer only.
    elif node_weights_q_cfg is not None and node_weights_q_cfg.has_weights_quantization_params():
        if node_weights_q_cfg.enable_weights_quantization:
            return keras.quantizer.WeightQuantizeConfigKD([KERNEL_NAME[n.layer_class]],
                                                          node_weights_q_cfg.weights_quantization_params.get(THRESHOLD),
                                                          node_weights_q_cfg.weights_channels_axis,
                                                          node_weights_q_cfg.weights_n_bits)
        else:
            return NoOpQuantizeConfig()

    # If the node should be not be quantized, use a QuantizeConfig which does not quantize.
    elif n.no_quantization():
        return NoOpQuantizeConfig()

    else:
        raise Exception('Undefined quantization method')

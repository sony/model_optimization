# ===============================================================================
# Copyright (c) 2021, Sony Semiconductors Israel, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ===============================================================================


from tensorflow.keras.layers import DepthwiseConv2D, Conv2D, Conv2DTranspose
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit.default_8bit_quantize_configs import \
    NoOpQuantizeConfig
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit.default_8bit_quantize_registry import \
    QuantizeConfig

from sony_model_optimization_package import common
from sony_model_optimization_package import keras
from sony_model_optimization_package.common.constants import THRESHOLD
from sony_model_optimization_package.keras.constants import DEPTHWISE_KERNEL, KERNEL

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

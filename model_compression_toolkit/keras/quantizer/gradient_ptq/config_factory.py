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
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit.default_8bit_quantize_configs import \
    NoOpQuantizeConfig
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit.default_8bit_quantize_registry import \
    QuantizeConfig

from model_compression_toolkit import common
from model_compression_toolkit import keras
from model_compression_toolkit.common.constants import THRESHOLD
from model_compression_toolkit.keras.constants import DEPTHWISE_KERNEL, KERNEL
from model_compression_toolkit.common.framework_info import FrameworkInfo


MAX_LSBS_CHANGE = 8


def quantization_config_builder_gptq(n: common.BaseNode,
                                     fw_info: FrameworkInfo) -> QuantizeConfig:
    """
    Build a QuantizeConfig for a node according to its quantization configuration and
    a global NoOpQuantizeConfig object.

    Args:
        n: Node to build its QuantizeConfig.
        fw_info: Framework information (e.g., mapping from layers to their attributes to quantize).

    Returns:
        A QuantizeConfig object with the appropriate quantizers (according to the node's
        quantization configuration).
    """

    if n.activation_quantization() and n.weight_quantization():
        qc = keras.quantizer.gradient_ptq.ActivationAndWeightQuantizeConfig(fw_info.get_kernel_op_attributes(n.layer_class),
                                                                            n.final_weights_quantization_cfg.weights_quantization_params.get(THRESHOLD),
                                                                            n.final_weights_quantization_cfg.weights_channels_axis,
                                                                            n.final_weights_quantization_cfg.weights_n_bits,
                                                                            n.activation_quantization_cfg.activation_quantization_params,
                                                                            n.activation_quantization_cfg.activation_is_signed,
                                                                            activation_num_bits=n.activation_quantization_cfg.activation_n_bits,
                                                                            max_lsbs_change=MAX_LSBS_CHANGE
                                                                            )
    elif n.activation_quantization():
        qc = keras.quantizer.gradient_ptq.ActivationQuantizeConfig(n.activation_quantization_cfg.activation_quantization_params,
                                                                   n.activation_quantization_cfg.activation_is_signed,
                                                                   num_bits=n.activation_quantization_cfg.activation_n_bits)
    elif n.weight_quantization():
        qc = keras.quantizer.gradient_ptq.WeightQuantizeConfig(fw_info.get_kernel_op_attributes(n.layer_class),
                                                               n.final_weights_quantization_cfg.weights_quantization_params.get(THRESHOLD),
                                                               n.final_weights_quantization_cfg.weights_channels_axis,
                                                               n.final_weights_quantization_cfg.weights_n_bits,
                                                               max_lsbs_change=MAX_LSBS_CHANGE
                                                               )

    elif n.no_quantization():
        qc = NoOpQuantizeConfig()

    else:
        raise Exception('Undefined quantization method')

    return qc

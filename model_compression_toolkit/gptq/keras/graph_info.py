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


import tensorflow as tf
from typing import Tuple, List

from model_compression_toolkit.core.keras.constants import USE_BIAS
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from tensorflow.keras.models import Model

from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.gptq.common.gptq_graph import get_kernel_attribute_name_for_gptq
from model_compression_toolkit.quantizers_infrastructure import KerasQuantizationWrapper


def get_gptq_trainable_parameters(fxp_model: Model,
                                  fw_info: FrameworkInfo,
                                  add_bias: bool = False) -> (
        List[tf.Variable], List[tf.Variable], List[tf.Variable], List[tf.Variable], List[tf.Variable]):
    """
    Get trainable parameters from all layers in a model

    Args:
        fxp_model: Model to get its trainable parameters.
        fw_info: Framework information needed for keras kernel ops list.
        add_bias: Whether to include biases of the model (if there are) or not.

    Returns:
        A list of trainable variables in a model. Each item is a list of a layers weights.
    """

    trainable_weights: List[tf.Tensor] = []
    trainable_threshold: List[tf.Tensor] = []
    bias_weights: List[List[tf.Tensor]] = []
    temperature_weights: List[tf.Tensor] = []

    for layer in fxp_model.layers:
        if isinstance(layer, KerasQuantizationWrapper):
            kernel_attribute = get_kernel_attribute_name_for_gptq(layer_type=type(layer.layer),
                                                                  fw_info=DEFAULT_KERAS_INFO)

            # collect trainable weights per layer
            layer_trainable_weights = layer.weights_quantizers[kernel_attribute].get_aux_variable()
            layer_trainable_threshold = layer.weights_quantizers[kernel_attribute].get_quantization_variable()

            if add_bias:
                kernel_ops_attrs = fw_info.kernel_ops_attributes_mapping.get(type(layer.layer))
                use_bias = kernel_ops_attrs is not None and kernel_ops_attrs[0] is not None \
                           and layer.layer.get_config().get(USE_BIAS)
                if use_bias is not None and use_bias:
                    bias_weights.append([layer.layer.bias])
            trainable_weights.append(layer_trainable_weights)
            trainable_threshold.extend(layer_trainable_threshold)

    return trainable_weights, bias_weights, trainable_threshold, temperature_weights


def get_weights_for_loss(fxp_model: Model) -> Tuple[List[list], List[list]]:
    """
    Get all float and quantized kernels for the GPTQ loss

    Args:
        fxp_model: Model to get its float and quantized weights.

    Returns:
        A list of float kernels, each item is the float kernel of the layer
        A list of quantized kernels, each item is the quantized kernel of the layer
    """

    flp_weights_list = []
    fxp_weights_list = []
    for layer in fxp_model.layers:
        if isinstance(layer, KerasQuantizationWrapper):

            # collect pairs of float and quantized weights per layer
            _layer_flp_weights, _layer_fxp_weights = [], []
            for weight, quantizer_vars, quantizer in layer.get_weights_vars():
                _layer_flp_weights.append(quantizer_vars)
                _layer_fxp_weights.append(quantizer(training=False, inputs=quantizer_vars))

            flp_weights_list.append(_layer_flp_weights)
            fxp_weights_list.append(_layer_fxp_weights)

    return flp_weights_list, fxp_weights_list


# TODO: this function need to move to location that is relevant only for soft quantizer -
#  once deciding how to handle GPTQ quantizers regularization.
def get_soft_rounding_reg(fxp_model: Model) -> List[tf.Tensor]:
    """
    This function returns the soft quantizer regularization values for SoftRounding.

    Args:
        fxp_model: A model to be quantized with SoftRounding.

    Returns: A list of tensors.
    """

    soft_reg_aux: List[tf.Tensor] = []
    for layer in fxp_model.layers:
        if isinstance(layer, KerasQuantizationWrapper):
            kernel_attribute = get_kernel_attribute_name_for_gptq(layer_type=type(layer.layer),
                                                                  fw_info=DEFAULT_KERAS_INFO)

            soft_reg_aux.append(layer.weights_quantizers[kernel_attribute].get_regularization())
    return soft_reg_aux

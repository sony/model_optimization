# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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
import torch
import torch.nn as nn
from typing import List
from model_compression_toolkit.core.pytorch.constants import BIAS
from model_compression_toolkit.core.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO
from model_compression_toolkit.gptq.common.gptq_graph import get_kernel_attribute_name_for_gptq
from model_compression_toolkit.logger import Logger
from mct_quantizers import PytorchQuantizationWrapper
from model_compression_toolkit.trainable_infrastructure.common.base_trainable_quantizer import VariableGroup


def get_gptq_trainable_parameters(fxp_model: nn.Module,
                                  add_bias: bool = False,
                                  ) -> (List[nn.Parameter], List[nn.Parameter], List[nn.Parameter]):
    """
    Get trainable parameters from all layers in a model

    Args:
        fxp_model: Model to get its trainable parameters.
        add_bias: Whether to include biases of the model (if there are) or not.

    Returns:
        A list of trainable variables in a model. Each item is a list of a layers weights.
    """

    trainable_aux_weights = nn.ParameterList()
    trainable_threshold = nn.ParameterList()
    trainable_bias = nn.ParameterList()

    for layer in fxp_model.modules():
        if isinstance(layer, PytorchQuantizationWrapper):
            kernel_attribute = get_kernel_attribute_name_for_gptq(layer_type=type(layer.layer),
                                                                  fw_info=DEFAULT_PYTORCH_INFO)

            # collect trainable weights per quantizer
            if kernel_attribute not in layer.weights_quantizers:
                Logger.critical(f"'{kernel_attribute}' was not found in the weight quantizers of layer '{layer.layer}'.")
            quantizer_trainable_weights = layer.weights_quantizers[kernel_attribute].get_trainable_variables(VariableGroup.WEIGHTS)
            quantizer_trainable_threshold = layer.weights_quantizers[kernel_attribute].get_trainable_variables(VariableGroup.QPARAMS)
            trainable_aux_weights.extend(quantizer_trainable_weights)
            trainable_threshold.extend(quantizer_trainable_threshold)

            if add_bias and hasattr(layer.layer, BIAS):
                bias = getattr(layer.layer, BIAS)
                if bias is not None:
                    trainable_bias.append(bias)

    return trainable_aux_weights, trainable_bias, trainable_threshold


def get_weights_for_loss(fxp_model: nn.Module) -> [List[nn.Parameter], List[torch.Tensor]]:
    """
    Get all float and quantized kernels for the GPTQ loss

    Args:
        fxp_model: Model to get its float and quantized weights.

    Returns:
        A list of float kernels, each item is the float kernel of the layer
        A list of quantized kernels, each item is the quantized kernel of the layer
    """

    flp_weights_list, fxp_weights_list = [], []
    for layer in fxp_model.modules():
        if isinstance(layer, PytorchQuantizationWrapper):
            # Collect pairs of float and quantized weights per layer
            for weight, quantizer_vars, quantizer in layer.get_weights_vars():
                flp_weights_list.append(quantizer_vars)
                fxp_weights_list.append(quantizer(training=False, inputs=quantizer_vars))

    return flp_weights_list, fxp_weights_list

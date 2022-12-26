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
from model_compression_toolkit.gptq.pytorch.quantizer.quantizer_wrapper import WeightQuantizerWrapper
from model_compression_toolkit.gptq.pytorch.quantizer.gumbel_rounding.base_gumbel_weights_quantizer import BaseGumbelWeightQuantizer
from model_compression_toolkit.core.pytorch.constants import BIAS


def get_trainable_parameters(fxp_model: nn.Module,
                             add_bias: bool = False,
                             quantization_parameters_learning: bool = False,
                             is_gumbel: bool = False) -> (List[nn.Parameter], List[nn.Parameter], List[nn.Parameter]):
    """
    Get trainable parameters from all layers in a model

    Args:
        fxp_model: Model to get its trainable parameters.
        add_bias: Whether to include biases of the model (if there are) or not.
        quantization_parameters_learning: Whether to include quantization parameters of the model or not.
        is_gumbel: Whether the fxp model is quantized using Gumbel Rounding
    Returns:
        A list of trainable variables in a model. Each item is a list of a layers weights.
    """

    trainable_aux_weights = nn.ParameterList()
    trainable_threshold = nn.ParameterList()
    trainable_bias = nn.ParameterList()
    trainable_temperature = nn.ParameterList()

    for layer in fxp_model.modules():
        if isinstance(layer, WeightQuantizerWrapper):
            trainable_aux_weights.append(layer.weight_quantizer.get_aux_variable())
            if quantization_parameters_learning:
                trainable_threshold.extend(layer.weight_quantizer.get_quantization_variable())
            if is_gumbel:
                trainable_temperature.append(layer.weight_quantizer.get_temperature_variable())
            if add_bias and hasattr(layer.op, BIAS):
                bias = getattr(layer.op, BIAS)
                trainable_bias.append(bias)

    return trainable_aux_weights, trainable_bias, trainable_threshold, trainable_temperature


def get_gumbel_probability(fxp_model: nn.Module) -> List[torch.Tensor]:
    """
    This function return the gumbel softmax probability of GumRounding
    Args:
        fxp_model: A model to be quantized with GumRounding

    Returns: A list of tensors.

    """
    gumbel_prob_aux = []
    for layer in fxp_model.modules():
        if isinstance(layer, WeightQuantizerWrapper) and isinstance(layer.weight_quantizer, BaseGumbelWeightQuantizer):
            gumbel_prob_aux.append(layer.weight_quantizer.get_gumbel_probability())
    return gumbel_prob_aux


def get_weights_for_loss(fxp_model: nn.Module) -> [List, List]:
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
        if isinstance(layer, WeightQuantizerWrapper):
            # Collect pairs of float and quantized weights per layer
            weights = layer.op.weight
            flp_weights_list.append(weights)
            fxp_weights_list.append(layer.weight_quantizer(weights, training=False))

    return flp_weights_list, fxp_weights_list

# Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
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
from typing import Dict, Any

import torch

from mct_quantizers import PytorchQuantizationWrapper
from mct_quantizers.common.constants import ACTIVATION_HOLDER_QUANTIZER
from model_compression_toolkit.core.pytorch.constants import BIAS


def get_layers_from_model_by_type(model: torch.nn.Module,
                                  layer_type: type,
                                  include_wrapped_layers: bool = True):
    """
    Return a list of layers of some type from a Pytorch module. The order of the returned list
    is according the order of the layers in model.named_children().
    If include_wrapped_layers is True, layers from that type that are wrapped using PytorchQuantizationWrapper
    are returned as well.

    Args:
        model: Pytorch model to get its layers.
        layer_type: Type of the layer we want to retrieve from the model.
        include_wrapped_layers: Whether to include layers that are wrapped using PytorchQuantizationWrapper.

    Returns:
        List of layers of type layer_type from the model.
    """
    if include_wrapped_layers:
        return [layer[1] for layer in model.named_children() if type(layer[1])==layer_type or (isinstance(layer[1], PytorchQuantizationWrapper) and type(layer[1].layer)==layer_type)]
    return [layer[1] for layer in model.named_children() if type(layer[1])==layer_type]


def count_model_prunable_params(model: torch.nn.Module) -> int:
    total_model_params = sum(p.numel() for p in model.state_dict().values())
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            total_model_params -= 1
    return total_model_params


def count_model_prunable_params(model: torch.nn.Module) -> int:
    """
    Calculates the total number of prunable parameters in a given PyTorch model.

    This function iterates over all parameters of the model, summing up their total count.
    It then adjusts this total by excluding parameters from certain layers that are deemed non-prunable.

    Args:
        model (torch.nn.Module): The PyTorch model to evaluate.

    Returns:
        int: The total count of prunable parameters in the model.
    """
    # Sum all parameters in the model to get the initial count.
    total_model_params = sum(p.numel() for p in model.state_dict().values())

    # Subtract non-prunable parameters, specifically from BatchNorm2d layers.
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            # Assuming one non-prunable ('num_batches_tracked') parameter per BatchNorm2d layer.
            total_model_params -= 1

    return total_model_params

def extract_model_weights(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Traverses a given PyTorch model and extracts the weights from all its layers,
    storing them in a dictionary. This function is specifically designed to handle
    both standard layers and layers wrapped in a PytorchQuantizationWrapper to get the
    quantized weights.

    Args:
        model (torch.nn.Module): The PyTorch model from which weights are to be extracted.

    Returns:
        dict: A dictionary containing the weights of the model.
    """

    weights_dict = {}
    visited_modules = set()

    for name, module in model.named_modules():
        if module in visited_modules:
            continue  # Skip already visited modules to avoid redundancy of inner layers (that are wrapped using PytorchQuantizationWrapper)

        if isinstance(module, PytorchQuantizationWrapper):
            # Extract quantized weights and optionally the bias
            q_weights = module.get_quantized_weights()
            if hasattr(module.layer, BIAS):
                q_weights[BIAS] = module.layer.bias

            # Update the weights dictionary with the quantized weights
            weights_dict.update({f"{name}.{k}": v for k, v in q_weights.items()})

            # Mark the inner layer as visited
            visited_modules.add(module.layer)
        else:
            # For other modules, directly add their parameters
            weights_dict.update({f"{name}.{param_name}" if name else param_name: param.data
                                 for param_name, param in module.named_parameters(recurse=False)})

    return weights_dict


def get_layer_type_from_activation_quantizer(model, layer_name):
    """
    Retrieves the type of layer corresponding to the given activation quantizer layer name from the model.

    Args:
        model (torch.nn.Module): The activation quantization wrapper containing the layer.
        layer_name (str): The name of the activation quantizer layer.

    Returns:
        torch.nn.Module: The layer associated with the activation quantizer. If the layer is wrapped in a
                         PytorchQuantizationWrapper, the inner layer is returned; otherwise, the layer itself is returned.
    """
    # Extract layer name
    layer_for_act_quant_name = layer_name.split('_' + ACTIVATION_HOLDER_QUANTIZER)[0]
    for name, layer in model.named_modules():
        if name == layer_for_act_quant_name:
            if isinstance(layer, PytorchQuantizationWrapper):
                return layer.layer  # Return the inner layer if wrapped
            else:
                return layer  # Return the layer
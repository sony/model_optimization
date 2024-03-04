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
import torch

from mct_quantizers import PytorchQuantizationWrapper


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
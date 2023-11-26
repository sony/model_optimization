# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
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
from typing import Dict, Callable

import torch
from torch import Tensor

from model_compression_toolkit.core.pytorch.utils import get_working_device
from model_compression_toolkit.data_generation.common.enums import OutputLossType
from model_compression_toolkit.data_generation.pytorch.model_info_exctractors import PytorchActivationExtractor

def min_max_diff(
        output_imgs: Tensor,
        activation_extractor: PytorchActivationExtractor,
        eps: float = 1e-6) -> Tensor:
    """
    Calculate the minimum-maximum difference of output images.

    Args:
        output_imgs (Tensor or List[Tensor]): The output of the model on images.
        activation_extractor (PytorchActivationExtractor): The activation extractor for the model.
        eps (float): Small value for numerical stability.

    Returns:
        Tensor: The computed minimum-maximum difference loss.
    """
    if not isinstance(output_imgs, (list, tuple)):
        output_imgs = [output_imgs]
    output_loss = 0
    for output in output_imgs:
        output = torch.reshape(output, [output.shape[0], -1])
        output_loss += 1 / torch.mean(torch.max(output, 1)[0] - torch.min(output, 1)[0] + eps)
    return output_loss


def regularized_min_max_diff(
        output_imgs: Tensor,
        activation_extractor: PytorchActivationExtractor,
        eps: float = 1e-6) -> Tensor:
    """
    Calculate the regularized minimum-maximum difference of output images. We want to maximize
    the difference between the minimum and maximum values of the output, but also to regularize
    their values so that they don't exceed the norm of the last layer's input times the norm of
    the last layer's weights.

    Args:
        output_imgs (Tensor or List[Tensor]): The output of the model on images.
        activation_extractor (PytorchActivationExtractor): The activation extractor for the model.
        eps (float): Small value for numerical stability.

    Returns:
        Tensor: The computed minimum-maximum difference loss.
    """
    # get the input to the last linear layers of the model
    output_layers_inputs = activation_extractor.get_output_layer_input_activation()

    # get the weights of the last linear layers of the model
    weights_output_layers = activation_extractor.get_last_linear_layers_weights()

    if not isinstance(output_imgs, (list, tuple)):
        output_imgs = torch.reshape(output_imgs, [output_imgs.shape[0], output_imgs.shape[1], -1])
        output_imgs = torch.mean(output_imgs, dim=-1)
        output_imgs = [output_imgs]
    output_loss = torch.zeros(1).to(get_working_device())

    for output_weight, output, last_layer_input in zip(weights_output_layers, output_imgs, output_layers_inputs):
        weights_norm = torch.linalg.norm(output_weight.squeeze(), dim=1)
        out_max, out_argmax = torch.max(output, dim=1)
        out_min, out_argmin = torch.min(output, dim=1)
        last_layer_avg = torch.reshape(last_layer_input, [last_layer_input.shape[0], last_layer_input.shape[1], -1])
        last_layer_avg = torch.mean(last_layer_avg, dim=-1)
        last_layer_norm = torch.linalg.norm(last_layer_avg, dim=1)
        reg_min = torch.abs(torch.abs(out_min) - 0.5 * last_layer_norm * weights_norm[out_argmin])
        reg_max = torch.abs(torch.abs(out_max) - 0.5 * last_layer_norm * weights_norm[out_argmax])
        dynamic_loss = 1 / (out_max - out_min + eps)
        output_loss += torch.mean(reg_min + reg_max + dynamic_loss)
    return output_loss

def no_output_loss(
        output_imgs: Tensor,
        activation_extractor: PytorchActivationExtractor,
        eps: float = 1e-6) -> Tensor:
    """
    Calculate no output loss.

    Args:
        output_imgs (Tensor): The output of the model on images.
        activation_extractor (PytorchActivationExtractor): The activation extractor for the model.
        eps (float): Small value for numerical stability.

    Returns:
        Tensor: A tensor with zero value for the loss.
    """
    return torch.zeros(1).to(get_working_device())


# Dictionary of output loss functions
output_loss_function_dict: Dict[OutputLossType, Callable] = {
    OutputLossType.NONE: no_output_loss,
    OutputLossType.MIN_MAX_DIFF: min_max_diff,
    OutputLossType.REGULARIZED_MIN_MAX_DIFF: regularized_min_max_diff,
}
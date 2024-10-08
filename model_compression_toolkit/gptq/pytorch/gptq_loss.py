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
from typing import List

import torch


def mse_loss(y: torch.Tensor, x: torch.Tensor, normalized: bool = True) -> torch.Tensor:
    """
    Compute the MSE of two tensors.
    Args:
        y: First tensor.
        x: Second tensor.
        normalized: either return normalized MSE or MSE
    Returns:
        The MSE of two tensors.
    """
    loss = torch.nn.MSELoss()(x, y)
    return loss / torch.mean(torch.square(x)) if normalized else loss


def multiple_tensors_mse_loss(y_list: List[torch.Tensor],
                              x_list: List[torch.Tensor],
                              fxp_w_list: List[List[torch.Tensor]],
                              flp_w_list: List[List[torch.Tensor]],
                              act_bn_mean: List,
                              act_bn_std: List,
                              loss_weights: torch.Tensor = None) -> torch.Tensor:
    """
    Compute MSE similarity between two lists of tensors

    Args:
        y_list: First list of tensors.
        x_list: Second list of tensors.
        fxp_w_list: list of lists each containing a quantized model layer's trainable weights - quantized
        flp_w_list: list of lists each containing a float model layer's weights - not quantized
        act_bn_mean: list of prior activations mean collected from batch normalization. None is there's no info
        act_bn_std: list of prior activations std collected from batch normalization. None is there's no info
        loss_weights: A vector of weights to compute weighted average loss.
    Returns:
        A single loss value which is the average of all MSE loss of all tensor pairs
        List of MSE similarities per tensor pair
    """

    loss_values_list = []
    for i, (y, x) in enumerate(zip(y_list, x_list)):
        point_loss = mse_loss(y, x)
        loss_values_list.append(point_loss)

    if loss_weights is not None:
        return torch.mean(loss_weights * torch.stack(loss_values_list))
    else:
        return torch.mean(torch.stack(loss_values_list))


def sample_layer_attention_loss(y_list: List[torch.Tensor],
                                x_list: List[torch.Tensor],
                                fxp_w_list,
                                flp_w_list,
                                act_bn_mean,
                                act_bn_std,
                                loss_weights: torch.Tensor) -> torch.Tensor:
    """
    Compute Sample Layer Attention loss between two lists of tensors.

    Args:
        y_list: First list of tensors.
        x_list: Second list of tensors.
        fxp_w_list, flp_w_list, act_bn_mean, act_bn_std: unused (needed to comply with the interface).
        loss_weights: layer-sample weights tensor of shape (batch X layers)

    Returns:
        Sample Layer Attention loss (a scalar).
    """
    loss = 0
    layers_mean_w = []

    for i, (y, x) in enumerate(zip(y_list, x_list)):
        norm = (y - x).pow(2).sum(1)
        if len(norm.shape) > 1:
            norm = norm.flatten(1).mean(1)
        w = loss_weights[:, i]
        loss += torch.mean(w * norm)
        layers_mean_w.append(w.mean())

    loss = loss / torch.stack(layers_mean_w).max()
    return loss


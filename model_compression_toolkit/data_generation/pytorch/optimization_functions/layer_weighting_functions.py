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

from model_compression_toolkit.data_generation.common.enums import LayerWeightingType
from model_compression_toolkit.data_generation.pytorch.model_info_exctractors import OrigBNStatsHolder, \
    ActivationExtractor


def average_layer_weighting_fn(orig_bn_stats_holder: OrigBNStatsHolder, **kwargs) -> Dict[str, float]:
    """
    Calculate average weighting for each batch normalization layer.

    Args:
        orig_bn_stats_holder (OrigBNStatsHolder): Holder for original batch normalization statistics.
        **kwargs: Additional arguments if needed.

    Returns:
        Dict[str, float]: A dictionary containing layer names as keys and average weightings as values.
    """
    num_bn_layers = orig_bn_stats_holder.get_num_bn_layers()
    return {bn_layer_name: 1 / num_bn_layers for bn_layer_name in orig_bn_stats_holder.get_bn_layer_names()}

def first_bn_multiplier_weighting_fn(orig_bn_stats_holder: OrigBNStatsHolder, **kwargs) -> Dict[str, float]:
    """
    Calculate layer weightings with a higher multiplier for the first batch normalization layer.

    Args:
        orig_bn_stats_holder (OrigBNStatsHolder): Holder for original batch normalization statistics.
        **kwargs: Additional arguments if needed.

    Returns:
        Dict[str, float]: A dictionary containing layer names as keys and weightings as values.
    """
    layer_weighting_dict = {orig_bn_stats_holder.get_bn_layer_names()[0]: 10}
    return layer_weighting_dict.update({bn_layer_name: 1  for bn_layer_name in orig_bn_stats_holder.get_bn_layer_names()[1:]})

def grad_layer_weighting_fn(orig_bn_stats_holder: OrigBNStatsHolder,
                               activation_extractor: ActivationExtractor,
                               activation_extractor_noised: ActivationExtractor,
                               **kwargs) -> Dict[str, float]:
    """
    Calculate layer weightings based on the gradient between activations and noised activations.

    Args:
        orig_bn_stats_holder (OrigBNStatsHolder): Holder for original batch normalization statistics.
        activation_extractor (ActivationExtractor): Extractor for layer activations.
        activation_extractor_noised (ActivationExtractor): Extractor for noised layer activations.
        **kwargs: Additional arguments if needed.

    Returns:
        Dict[str, float]: A dictionary containing layer names as keys and weightings as values.
    """
    eps = 1e-6
    layer_weighting_dict = {}
    num_bn_layers = orig_bn_stats_holder.get_num_bn_layers()
    for layer in activation_extractor.get_hooked_layer_names():
        activation = activation_extractor.get_activation(layer)
        activation_noised = activation_extractor_noised.get_activation(layer)
        delta_grad = torch.linalg.norm(
            torch.mean(activation.detach() - activation_noised.detach(), dim=(0, 2, 3))) + eps
        layer_weighting_dict.update({layer: 1 / delta_grad})
    return layer_weighting_dict


# Dictionary of layer weighting functions
layer_weighting_function_dict: Dict[LayerWeightingType, Callable] = {
    LayerWeightingType.AVERAGE: average_layer_weighting_fn,
    LayerWeightingType.FIRST_LAYER_MULTIPLIER: first_bn_multiplier_weighting_fn,
    # LayerWeightingType.GRAD: grad_layer_weighting_fn
}

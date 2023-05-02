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
import copy
from typing import Any, Callable

import torch
from tqdm import tqdm

from model_compression_toolkit.core import CoreConfig
from model_compression_toolkit.core import common
from model_compression_toolkit.core.pytorch.constants import GAMMA, BETA, MOVING_MEAN, MOVING_VARIANCE
from model_compression_toolkit.core.pytorch.utils import set_model, to_torch_tensor


def pytorch_apply_second_moment_correction(quantized_model: Any,
                                           core_config: CoreConfig,
                                           representative_data_gen: Callable,
                                           graph: common.Graph):
    """
    Apply second moment statistics correction to graph.

    Args:
        quantized_model: Framework's model to apply second moment correction on.
        core_config: QuantizationConfig of how the model should be quantized.
        representative_data_gen: Dataset to use for retrieving images for the models inputs.
        graph: Graph to update the parameters after the second moment correction.

    Returns:
        A function that applies second moment correction.
    """
    model = copy.deepcopy(quantized_model)
    set_model(model)

    # Move every BN to train mode
    for name, module in model.named_modules():
        if len(graph.find_node_by_name(name)) > 0:
            node = graph.find_node_by_name(name)[0]
            if isinstance(module, torch.nn.BatchNorm2d) and node.final_weights_quantization_cfg\
                    .weights_second_moment_correction:
                module.train()

    with torch.no_grad():
        for data in tqdm(representative_data_gen()):
            model(*to_torch_tensor(data))

    set_model(model)

    # Move every BN to eval mode and update the corresponding BN node params in the graph
    for name, module in model.named_modules():
        if len(graph.find_node_by_name(name)) > 0:
            node = graph.find_node_by_name(name)[0]
            if isinstance(module, torch.nn.BatchNorm2d) and node.final_weights_quantization_cfg\
                    .weights_second_moment_correction:
                module.eval()
                bn_node_weights = {GAMMA: module.weight.detach().cpu().numpy(),
                                   BETA: module.bias.detach().cpu().numpy(),
                                   MOVING_MEAN: module.running_mean.detach().cpu().numpy(),
                                   MOVING_VARIANCE: module.running_var.detach().cpu().numpy()}
                node.weights = copy.deepcopy(bn_node_weights)

    return graph


#  Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==============================================================================
#
from typing import Callable, Any

from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.network_editors.edit_network import edit_network_graph
from model_compression_toolkit.core.common.framework_info import FrameworkInfo

from xquant import XQuantConfig


def edit_quantized_graph(quantized_graph: Graph,
                         fw_info: FrameworkInfo,
                         xquant_config: XQuantConfig,
                         back2fw_fn: Callable) -> Any:
    """
    Edit the quantized graph according to the provided configuration and framework information.

    Args:
        quantized_graph (Graph): The graph representation of the quantized model.
        fw_info (FrameworkInfo): Information specific to the framework being used.
        xquant_config (XQuantConfig): Configuration settings for explainable quantization, including edit rules.
        back2fw_fn (Callable): A function that converts the edited graph back to the framework's model format.

    Returns:
        Any: The edited quantized model.
    """

    # Edit the quantized graph based on the edit rules specified in the xquant_config.
    edit_network_graph(quantized_graph,
                       fw_info,
                       xquant_config.edit_rules)

    # Convert the edited graph back to the framework-specific model format using the provided function.
    edited_quantized_model = back2fw_fn(quantized_graph)

    return edited_quantized_model


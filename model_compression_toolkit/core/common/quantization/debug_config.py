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

from dataclasses import dataclass, field
from typing import List

from model_compression_toolkit.core.common.network_editors.edit_network import EditRule


@dataclass
class DebugConfig:
    """
    A dataclass for MCT core debug information.

    Args:
        analyze_similarity (bool): Whether to plot similarity figures within TensorBoard (when logger is
         enabled) or not. Can be used to pinpoint problematic layers in the quantization process.
        network_editor (List[EditRule]): A list of rules and actions to edit the network for quantization.
        simulate_scheduler (bool): Simulate scheduler behavior to compute operators' order and cuts.
    """

    analyze_similarity: bool = False
    network_editor: List[EditRule] = field(default_factory=list)
    simulate_scheduler: bool = False

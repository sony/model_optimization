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
import torch
from torch.nn import Module

from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.pytorch.utils import set_model


def node_builder(n: BaseNode) -> Module:
    """
    Build a Pytorch module from a node.

    Args:
        n: Node to build its Pytorch layer

    Returns:
        Pytorch module that was built from the node.
    """

    framework_attr = copy.copy(n.framework_attr)
    node_instance = n.type(**framework_attr)
    node_instance.load_state_dict({k: torch.Tensor(v) for k, v in n.weights.items()}, strict=False)
    set_model(node_instance)
    return node_instance
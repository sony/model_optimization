# Copyright 2022 Sony Semiconductors Israel, Inc. All rights reserved.
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

from typing import Any, Dict

from model_compression_toolkit.core.common.graph.base_node import BaseNode
from model_compression_toolkit.core.common.target_platform.targetplatform2framework.attribute_filter import AttributeFilter


class LayerFilterParams:
    """
    Wrap a layer with filters to filter framework's layers by their attributes.
    """

    def __init__(self, layer,
                 *conditions,
                 **kwargs):
        """

        Args:
            layer: Layer to match when filtering.
            *conditions (AttributeFilter): List of conditions to satisfy.
            **kwargs: Keyword arguments to filter layers according to.
        """
        self.layer = layer
        self.conditions = conditions
        self.kwargs = kwargs
        self.__name__ = self.create_name()

    def __hash__(self):
        """

        Returns: Hash code for the the LayerFilterParams. Used to check if a LayerFilterParams
        is mapped to multiple OperatorsSet.

        """
        return hash(self.__name__)  # TODO: reuven: will hash differently if conditions/kwargs are in different order. fix it

    def __eq__(self, other: Any) -> bool:
        """
        Check if an object is equal to the LayerFilterParams.

        Args:
            other: Object to check.

        Returns:
            Whether an object is equal to the LayerFilterParams.
        """

        if not isinstance(other, LayerFilterParams):
            return False

        # Check equality of conditions
        for self_c, other_c in zip(self.conditions, other.conditions):
            if self_c != other_c:
                return False

        # Check key-value arguments equality
        for k, v in self.kwargs.items():
            if k not in other.kwargs:
                return False
            else:
                if other.kwargs.get(k) != v:
                    return False
        return True

    def create_name(self) -> str:
        """

        Returns: Name of the LayerFilterParams. The name is composed of the layer type,
        conditions and keyword arguments. Used for display and hashing.

        """
        params = [f'{k}={v}' for k,v in self.kwargs.items()]
        params.extend([str(c) for c in self.conditions])
        params_str = ', '.join(params)
        return f'{self.layer.__name__}({params_str})'

    def match(self,
              node: BaseNode) -> bool:
        """
        Check if a node matches the layer, conditions and keyword-arguments of
        the LayerFilterParams.

        Args:
            node: Node to check if matches to the LayerFilterParams properties.

        Returns:
            Whether the node matches to the LayerFilterParams properties.
        """
        # Check the node has the same type as the layer in LayerFilterParams
        if self.layer != node.type:
            return False

        # Get attributes from node to filter
        layer_config = node.framework_attr
        if hasattr(node, "op_call_kwargs"):
            layer_config.update(node.op_call_kwargs)

        for attr, value in self.kwargs.items():
            if layer_config.get(attr) != value:
                return False

        for c in self.conditions:
            if not c.match(layer_config):
                return False

        return True

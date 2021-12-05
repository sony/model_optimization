# Copyright 2021 Sony Semiconductors Israel, Inc. All rights reserved.
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


from abc import ABC, abstractmethod
from typing import Any

from model_compression_toolkit.common.graph.base_graph import Graph
from model_compression_toolkit.common.matchers.base_matcher import BaseMatcher

class BaseSubstitution(ABC):
    """
    Base class for all substitution classes.
    """

    def __init__(self, matcher_instance: BaseMatcher):
        """
        Init for class BaseSubstitution.

        Args:
            matcher_instance: A BaseNodeMatcher object to store.
        """

        self.matcher_instance = matcher_instance

    @abstractmethod
    def substitute(self, graph: Graph, object_to_sub: Any):
        """
        Transform the graph after matching the object to substitute.

        Args:
            graph: Graph to apply the substitution on.
            object_to_sub: Matched object to substitute.

        Returns:
            Graph after the substitution is applied.

        """

        pass  # pragma: no cover

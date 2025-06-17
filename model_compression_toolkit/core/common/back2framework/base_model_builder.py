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
from abc import ABC, abstractmethod
from typing import Any, Tuple

from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common.user_info import UserInformation


class BaseModelBuilder(ABC):
    """
    Base class for model builder.
    """

    def __init__(self,
                 graph: common.Graph,
                 append2output=None,
                 return_float_outputs: bool = False):
        """

        Args:
            graph: Graph to build the model from.
            append2output: Nodes of graph to append to model's output.
            return_float_outputs: Whether the model returns float tensors or not.
        """

        self.graph = graph
        self.append2output = append2output
        self.return_float_outputs = return_float_outputs

    @abstractmethod
    def build_model(self) -> Tuple[Any, UserInformation]:
        """

        Returns: A framework's model built from its graph.

        """
        raise NotImplemented(f'{self.__class__.__name__} have to implement build_model method.')  # pragma: no cover

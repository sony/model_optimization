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
from typing import Any, Dict

from model_compression_toolkit.target_platform_capabilities.target_platform.current_tp_model import _current_tp_model


class TargetPlatformModelComponent:
    """
    Component of TargetPlatformModel (Fusing, OperatorsSet, etc.)
    """
    def __init__(self, name: str):
        """

        Args:
            name: Name of component.
        """
        self.name = name
        _current_tp_model.get().append_component(self)

    def get_info(self) -> Dict[str, Any]:
        """

        Returns: Get information about the component to display (return an empty dictionary.
        the actual component should fill it with info).

        """
        return {}

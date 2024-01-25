# Copyright 2021 Sony Semiconductor Israel, Inc. All rights reserved.
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


from typing import Dict, Any
from copy import deepcopy


class DefaultDict:
    """
    Default dictionary. It wraps a dictionary given at initialization and return its
    values when requested. If the requested key is not presented at initial dictionary,
    it returns the returned value a default value (that is passed at initialization) generates.
    """

    def __init__(self,
                 known_dict: Dict[Any, Any] = None,
                 default_value: Any = None):
        """

        Args:
            known_dict: Dictionary to wrap. If None is provided, initializes an empty dictionary.
            default_value: default value when requested key is not in known_dict.
        """

        self.default_value = default_value
        self.known_dict = known_dict if known_dict is not None else {}

    def get(self, key: Any) -> Any:
        """
        Get the value of the inner dictionary by the given key, If key is not in dictionary,
        it uses the default_factory to return a default value.

        Args:
            key: Key to use in inner dictionary.

        Returns:
            Value of the inner dictionary by the given key, or a default value if not exist.
            If default_factory was not passed at initialization, it returns None.
        """

        if key in self.known_dict:
            return self.known_dict[key]
        else:
            return deepcopy(self.default_value)

    def keys(self):
        """
        Get keys of known_dict
        Returns: keys of known_dict
        """
        return self.known_dict.keys()

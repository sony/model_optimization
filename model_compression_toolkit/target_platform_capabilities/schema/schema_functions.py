# Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
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
from typing import Any, Dict


def clone_and_edit_object_params(obj: Any, **kwargs: Dict) -> Any:
    """
    Clones the given object and edit some of its parameters.

    Args:
        obj: An object to clone.
        **kwargs: Keyword arguments to edit in the cloned object.

    Returns:
        Edited copy of the given object.
    """

    obj_copy = copy.deepcopy(obj)
    for k, v in kwargs.items():
        assert hasattr(obj_copy,
                       k), f'Edit parameter is possible only for existing parameters in the given object, ' \
                           f'but {k} is not a parameter of {obj_copy}.'
        setattr(obj_copy, k, v)
    return obj_copy

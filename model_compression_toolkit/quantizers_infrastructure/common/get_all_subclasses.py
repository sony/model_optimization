# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
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
from typing import Set


def get_all_subclasses(cls: type) -> Set[type]:
    """
    This function returns a list of all subclasses of the given class,
    including all subclasses of those subclasses, and so on.
    Recursively get all subclasses of the subclass and add them to the list of all subclasses.

    Args:
        cls: A class object.

    Returns: All classes that inherit from cls.

    """

    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in get_all_subclasses(c)])

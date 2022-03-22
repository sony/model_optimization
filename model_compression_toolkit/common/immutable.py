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
from typing import Any


class ImmutableClass(object):
    """
    Class to make inherits classes immutable.
    """
    _initialized = False

    def __init__(self):
        self._initialized = False

    def __setattr__(self,
                    *args: Any,
                    **kwargs: Any):
        """
        Use this method to enforce immutability when object is finalized.

        Args:
            *args: Arguments to set to attribute.
            **kwargs: Keyword-arguments to set to attribute.

        """
        if self._initialized:
            raise Exception('Immutable class. Can\'t edit attributes')
        else:
            object.__setattr__(self,
                               *args,
                               **kwargs)

    def initialized_done(self):
        """

        Method to use when object should be immutable.

        """
        if self._initialized:
            raise Exception('reinitialized')  # Can not get finalized again.
        self._initialized = True  # Finalize object.

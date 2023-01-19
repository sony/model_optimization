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


from abc import abstractmethod
from typing import Any


class BaseMatcher(object):
    """
    Base class for matchers. Matcher has the ability to check if an input object
    satisfies a condition the matcher checks.
    """

    @abstractmethod
    def apply(self, input_object: Any) -> bool:
        """
        Check if input_object satisfies a condition the matcher checks.
        Args:
            input_object: Object to check.

        Returns:
            True if input_object satisfies a condition the matcher checks.
            Otherwise, return nothing.
        """
        pass  # pragma: no cover

    def __and__(self, other: Any):
        """
        Return a matcher to check the logic AND of two BaseMatchers on an object.
        """
        raise NotImplemented  # pragma: no cover

    def __or__(self, other: Any):
        """
        Return a matcher to check the logic OR of BaseMatchers on an object.
        """
        raise NotImplemented  # pragma: no cover

    def logic_not(self):
        """
        Return a matcher to check the logic NOT of the BaseMatcher on an object.
        """
        raise NotImplemented  # pragma: no cover

    def logic_and(self, other: Any):
        """
        Return a matcher to check the logic AND of two BaseMatchers on an object.
        """
        return self & other

    def logic_or(self, other: Any):
        """
        Return a matcher to check the logic OR of two BaseMatchers on an object.
        """
        return self | other

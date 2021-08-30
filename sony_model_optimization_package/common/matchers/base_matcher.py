# ===============================================================================
# Copyright (c) 2021, Sony Semiconductors Israel, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ===============================================================================


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
        raise NotImplemented

    def __or__(self, other: Any):
        """
        Return a matcher to check the logic OR of BaseMatchers on an object.
        """
        raise NotImplemented

    def logic_not(self):
        """
        Return a matcher to check the logic NOT of the BaseMatcher on an object.
        """
        raise NotImplemented

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

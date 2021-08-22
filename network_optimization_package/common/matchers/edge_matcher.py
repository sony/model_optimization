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


from network_optimization_package.common.matchers.node_matcher import BaseNodeMatcher
from . import base_matcher


class BaseEdgeMatcher(base_matcher.BaseMatcher):

    def __init__(self, source_matcher: BaseNodeMatcher, target_matcher: BaseNodeMatcher):
        """
        Initialize a BaseEdgeMatcher object with source and target BaseNodeMatchers.

        Args:
            source_matcher: Source node matcher to check if matches the edge source.
            target_matcher: Source node matcher to check if matches the edge target.
        """
        self.source_matcher = source_matcher
        self.target_matcher = target_matcher

    def __and__(self, other):
        """
        Return a matcher to check the logic AND of two edge matchers on an object.
        """
        return EdgeAndMatcher(self, other)

    def __or__(self, other):
        """
        Return a matcher to check the logic AND of two edge matchers on an object.
        """
        return EdgeOrMatcher(self, other)

    def logic_not(self):
        """
        Return a matcher to check the logic NOT of an edge matchers on an object.
        """
        return EdgeNotMatcher(self)

    def apply(self, input_object) -> bool:
        """
        Check if input_object matches the matcher condition.

        Args:
            input_object: An edge object to check the matcher on.

        Returns:
            True if there is a match, otherwise False.
        """
        if isinstance(input_object, tuple) and len(input_object) >= 2:
            return self.source_matcher.apply(input_object[0]) and self.target_matcher.apply(input_object[1])
        else:
            return False


class EdgeAndMatcher(BaseEdgeMatcher):
    """
    An edge matcher to check the logic AND of two edge matchers on an object.
    """

    def __init__(self, matcher_a: BaseEdgeMatcher, matcher_b: BaseEdgeMatcher):
        self.matcher_a = matcher_a
        self.matcher_b = matcher_b

    def apply(self, input_object) -> bool:
        return self.matcher_a.apply(input_object) and self.matcher_b.apply(input_object)


class EdgeOrMatcher(BaseEdgeMatcher):
    """
    An edge matcher to check the logic OR of two edge matchers on an object.
    """

    def __init__(self, matcher_a: BaseEdgeMatcher, matcher_b: BaseEdgeMatcher):
        self.matcher_a = matcher_a
        self.matcher_b = matcher_b

    def apply(self, input_object) -> bool:
        return self.matcher_a.apply(input_object) or self.matcher_b.apply(input_object)


class EdgeAnyMatcher(BaseEdgeMatcher):
    """
    An edge matcher to check the logic AND of two edge matchers on an object.
    """

    def apply(self, input_object) -> bool:
        return True


class EdgeNotMatcher(BaseEdgeMatcher):
    """
    An edge matcher to check the logic NOT of an edge matcher on an object.
    """

    def __init__(self, matcher_a):
        self.matcher_a = matcher_a

    def apply(self, input_object) -> bool:
        return not self.matcher_a.apply(input_object)

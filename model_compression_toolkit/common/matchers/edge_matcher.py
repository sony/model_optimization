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


from model_compression_toolkit.common.matchers.node_matcher import BaseNodeMatcher
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

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


from . import base_matcher


class BaseNodeMatcher(base_matcher.BaseMatcher):
    """
    Base class for matchers that match a node in the graph.
    An extension of logics AND, OR and NOT are implemented.
    """

    def __and__(self, other):
        """
        Return a matcher to check the logic AND of two BaseNodeMatcher on an object.
        """
        return NodeAndMatcher(self, other)

    def __or__(self, other):
        """
        Return a matcher to check the logic OR of two BaseNodeMatcher on an object.
        """
        return NodeOrMatcher(self, other)

    def logic_not(self):
        """
        Return a matcher to check the logic NOT of the BaseNodeMatcher on an object.
        """
        return NodeNotMatcher(self)


class NodeAndMatcher(BaseNodeMatcher):
    """
    Matcher to check the logic AND of two node matchers on an object.
    """

    def __init__(self, matcher_a: BaseNodeMatcher, matcher_b: BaseNodeMatcher):
        self.matcher_a = matcher_a
        self.matcher_b = matcher_b

    def apply(self, input_object) -> bool:
        return self.matcher_a.apply(input_object) and self.matcher_b.apply(input_object)


class NodeOrMatcher(BaseNodeMatcher):
    """
    Matcher to check the logic OR of two node matchers on an object.
    """

    def __init__(self, matcher_a: BaseNodeMatcher, matcher_b: BaseNodeMatcher):
        self.matcher_a = matcher_a
        self.matcher_b = matcher_b

    def apply(self, input_object) -> bool:
        return self.matcher_a.apply(input_object) or self.matcher_b.apply(input_object)


class NodeAnyMatcher(BaseNodeMatcher):
    """
    Matcher to return always True, to any given input object.
    """

    def apply(self, input_object) -> bool:
        return True


class NodeNotMatcher(BaseNodeMatcher):
    """
    Matcher to check the logic AND of two node matchers on an object.
    """

    def __init__(self, matcher_a):
        self.matcher_a = matcher_a

    def apply(self, input_object) -> bool:
        return not self.matcher_a.apply(input_object)

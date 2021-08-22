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

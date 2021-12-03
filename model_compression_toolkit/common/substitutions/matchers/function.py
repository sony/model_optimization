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
from . import edge_matcher
from . import node_matcher
from . import walk_matcher


def is_node_matcher(matcher: base_matcher.BaseMatcher):
    """
    Check whether a matcher is of type node matcher or not.
    Args:
        matcher: Matcher to check.

    Returns:
        Whether a matcher is of type node matcher or not.
    """
    return issubclass(matcher.__class__, node_matcher.BaseNodeMatcher)


def is_edge_matcher(matcher: base_matcher.BaseMatcher):
    """
    Check whether a matcher is of type edge matcher or not.
    Args:
        matcher: Matcher to check.

    Returns:
        Whether a matcher is of type edge matcher or not.
    """
    return issubclass(matcher.__class__, edge_matcher.BaseEdgeMatcher)


def is_walk_matcher(matcher: base_matcher.BaseMatcher):
    """
    Check whether a matcher is of type walk matcher or not.
    Args:
        matcher: Matcher to check.

    Returns:
        Whether a matcher is of type walk matcher or not.
    """
    return issubclass(matcher.__class__, walk_matcher.WalkMatcherList)

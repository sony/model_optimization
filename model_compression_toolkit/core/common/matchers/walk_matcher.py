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

from abc import ABC

from . import base_matcher


class BaseWalkMatcher(base_matcher.BaseMatcher):
    """
    Base class for walk matchers.
    """
    pass  # pragma: no cover


class WalkMatcherList(BaseWalkMatcher, ABC):
    """
    Initialize a WalkMatcherList object with a list of nodes to match
    in a graph.
    """

    def __init__(self, matcher_list: list):
        self.matcher_list = matcher_list

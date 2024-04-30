# Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
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

import unittest

from model_compression_toolkit.core.common.matchers.edge_matcher import BaseEdgeMatcher, EdgeAndMatcher, EdgeOrMatcher, \
    EdgeNotMatcher, EdgeAnyMatcher


class NodeMatcherStub:
    def __init__(self, response):
        self.response = response

    def apply(self, input):
        return self.response

class TestEdgeMatcher(unittest.TestCase):
    def test_init(self):
        sm = NodeMatcherStub(True)
        tm = NodeMatcherStub(False)
        bem = BaseEdgeMatcher(sm, tm)
        self.assertIs(bem.source_matcher, sm)
        self.assertIs(bem.target_matcher, tm)

    def test_apply_correct_input(self):
        sm = NodeMatcherStub(True)
        tm = NodeMatcherStub(True)
        bem = BaseEdgeMatcher(sm, tm)
        self.assertTrue(bem.apply((1, 1)))

    def test_apply_incorrect_input(self):
        sm = NodeMatcherStub(True)
        tm = NodeMatcherStub(True)
        bem = BaseEdgeMatcher(sm, tm)
        self.assertFalse(bem.apply((1,)))  # Not enough elements

    def test_and(self):
        sm = NodeMatcherStub(True)
        tm = NodeMatcherStub(True)
        bem = BaseEdgeMatcher(sm, tm)
        result = bem.__and__(bem)
        self.assertIsInstance(result, EdgeAndMatcher)

    def test_or(self):
        sm = NodeMatcherStub(True)
        tm = NodeMatcherStub(True)
        bem = BaseEdgeMatcher(sm, tm)
        result = bem.__or__(bem)
        self.assertIsInstance(result, EdgeOrMatcher)

    def test_not(self):
        sm = NodeMatcherStub(True)
        tm = NodeMatcherStub(True)
        bem = BaseEdgeMatcher(sm, tm)
        result = bem.logic_not()
        self.assertIsInstance(result, EdgeNotMatcher)

    def test_edge_and_matcher_apply(self):
        sm = NodeMatcherStub(True)
        tm = NodeMatcherStub(False)
        bem1 = BaseEdgeMatcher(sm, tm)
        bem2 = BaseEdgeMatcher(tm, sm)
        eam = EdgeAndMatcher(bem1, bem2)
        self.assertFalse(eam.apply((1, 1)))

    def test_edge_or_matcher_apply(self):
        sm = NodeMatcherStub(True)
        tm = NodeMatcherStub(True)
        bem1 = BaseEdgeMatcher(sm, tm)

        sm2 = NodeMatcherStub(True)
        tm2 = NodeMatcherStub(False)
        bem2 = BaseEdgeMatcher(tm2, sm2)

        eom = EdgeOrMatcher(bem1, bem2)
        self.assertTrue(eom.apply((1, 1)))

    def test_edge_not_matcher_apply(self):
        sm = NodeMatcherStub(True)
        tm = NodeMatcherStub(False)
        bem = BaseEdgeMatcher(sm, tm)
        enm = EdgeNotMatcher(bem)
        self.assertTrue(enm.apply((1, 1)))

    def test_edge_any_matcher_apply(self):
        eam = EdgeAnyMatcher(None, None)
        self.assertTrue(eam.apply((1, 1)))

if __name__ == '__main__':
    unittest.main()

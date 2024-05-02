#  Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==============================================================================

import unittest
from unittest.mock import patch

from model_compression_toolkit.target_platform_capabilities.immutable import ImmutableClass


class TestImmutableClass(unittest.TestCase):

    def test_attribute_setting_before_finalization(self):
        obj = ImmutableClass()
        obj.some_attribute = "test_value"
        self.assertEqual(obj.some_attribute, "test_value")

    def test_attribute_setting_after_finalization(self):
        obj = ImmutableClass()
        obj.initialized_done()
        with self.assertRaises(Exception):
            obj.some_attribute = "new_value"

    @patch('model_compression_toolkit.logger.Logger')
    def test_error_logging_on_attribute_change_after_finalization(self, mock_logger):
        obj = ImmutableClass()
        obj.initialized_done()
        with self.assertRaises(Exception):
            obj.some_attribute = "new_value"
            mock_logger.critical.assert_called_once_with("Immutable class. Can't edit attributes.")

    @patch('model_compression_toolkit.logger.Logger')
    def test_finalization_only_once(self, mock_logger):
        obj = ImmutableClass()
        obj.initialized_done()  # First call to finalize
        with self.assertRaises(Exception):
            obj.initialized_done()  # Second call to finalize
            mock_logger.critical.assert_called_once_with('Object reinitialization error: object cannot be finalized again.')

    def test_multiple_finalizations(self):
        obj = ImmutableClass()
        obj.initialized_done()
        with self.assertRaises(Exception):
            obj.initialized_done()

if __name__ == '__main__':
    unittest.main()

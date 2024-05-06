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
from unittest.mock import patch, MagicMock
import logging
from io import StringIO

from model_compression_toolkit.logger import Logger, set_log_folder


class TestLogger(unittest.TestCase):
    def setUp(self):
        self.log_folder = "test_logs"
        self.log_level = logging.DEBUG
        self.log_message = "Test message"

    @patch('pathlib.Path.mkdir')
    @patch('os.path.exists')
    def test_check_path_create_dir(self, mock_exists, mock_mkdir):
        mock_exists.return_value = False
        Logger._Logger__check_path_create_dir(self.log_folder)  # Using the mangled name
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch('model_compression_toolkit.logger.Logger.get_logger')
    def test_set_logger_level(self, mock_get_logger):
        logger_mock = MagicMock()
        mock_get_logger.return_value = logger_mock
        Logger.set_logger_level(self.log_level)
        logger_mock.setLevel.assert_called_once_with(self.log_level)

    @patch('model_compression_toolkit.logger.logging.getLogger')
    def test_get_logger(self, mock_get_logger):
        Logger.get_logger()
        mock_get_logger.assert_called_once_with('Model Compression Toolkit')

    @patch('model_compression_toolkit.logger.Logger.get_logger')
    @patch('model_compression_toolkit.logger.logging.FileHandler')
    def test_set_log_file(self, mock_file_handler, mock_get_logger):
        logger_mock = MagicMock()
        mock_get_logger.return_value = logger_mock
        Logger.set_log_file(self.log_folder)
        mock_file_handler.assert_called_once()
        logger_mock.addHandler.assert_called_once()

    @patch('model_compression_toolkit.logger.logging.shutdown')
    def test_shutdown(self, mock_shutdown):
        Logger.shutdown()
        mock_shutdown.assert_called_once()
        self.assertIsNone(Logger.LOG_PATH)

    @patch('model_compression_toolkit.logger.Logger.get_logger')
    def test_critical(self, mock_get_logger):
        logger_mock = MagicMock()
        mock_get_logger.return_value = logger_mock
        with self.assertRaises(Exception) as context:
            Logger.critical(self.log_message)
        self.assertTrue(self.log_message in str(context.exception))
        logger_mock.critical.assert_called_once_with(self.log_message)

    @patch('model_compression_toolkit.logger.Logger.get_logger')
    def test_debug(self, mock_get_logger):
        logger_mock = MagicMock()
        mock_get_logger.return_value = logger_mock
        Logger.debug(self.log_message)
        logger_mock.debug.assert_called_once_with(self.log_message)

    @patch('model_compression_toolkit.logger.Logger.get_logger')
    def test_info(self, mock_get_logger):
        logger_mock = MagicMock()
        mock_get_logger.return_value = logger_mock
        with patch('sys.stdout', new=StringIO()) as fake_out:
            Logger.info(self.log_message)
            self.assertEqual(fake_out.getvalue().strip(), self.log_message)
        logger_mock.info.assert_called_once_with(self.log_message)

    @patch('model_compression_toolkit.logger.Logger.get_logger')
    def test_warning(self, mock_get_logger):
        logger_mock = MagicMock()
        mock_get_logger.return_value = logger_mock
        with patch('sys.stdout', new=StringIO()) as fake_out:
            Logger.warning(self.log_message)
            self.assertEqual(fake_out.getvalue().strip(), self.log_message)
        logger_mock.warning.assert_called_once_with(self.log_message)

    @patch('model_compression_toolkit.logger.Logger.get_logger')
    def test_error(self, mock_get_logger):
        logger_mock = MagicMock()
        mock_get_logger.return_value = logger_mock
        Logger.error(self.log_message)
        logger_mock.error.assert_called_once_with(self.log_message)

    @patch('model_compression_toolkit.logger.Logger.set_log_file')
    @patch('model_compression_toolkit.logger.Logger.set_logger_level')
    def test_set_log_folder(self, mock_set_logger_level, mock_set_log_file):
        set_log_folder(self.log_folder, self.log_level)
        mock_set_log_file.assert_called_once_with(self.log_folder)
        mock_set_logger_level.assert_called_once_with(self.log_level)


if __name__ == '__main__':
    unittest.main()


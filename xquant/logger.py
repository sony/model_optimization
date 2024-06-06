#  Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==============================================================================

import logging
import os
from typing import Optional


class Logger:
    _logger: Optional[logging.Logger] = None

    @staticmethod
    def get_logger(
            name: str = 'xquant',
            level: int = logging.INFO,
            log_dir: str = '.',
            log_file: str = 'xquant_log_file.log'
    ) -> logging.Logger:
        """
        Gets a logger instance with the specified name, logging level, and log file configuration.

        Args:
            name (str): Name of the logger. Defaults to 'xquant'.
            level (int): Logging level. Defaults to logging.INFO.
            log_dir (str): Directory where the log file will be stored. Defaults to current directory.
            log_file (str): Name of the log file. Defaults to 'xquant_log_file.log'.

        Returns:
            logging.Logger: Configured logger instance.
        """
        if Logger._logger is None:
            Logger._logger = logging.getLogger(name)
            Logger._logger.setLevel(level)

            # Ensure the directory exists
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            # Full path for the log file
            log_file_path = os.path.join(log_dir, log_file)

            # Create a file handler
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setLevel(level)

            # Create a formatter and set it for the handlers
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)

            # Add the handlers to the logger
            Logger._logger.addHandler(file_handler)

        return Logger._logger


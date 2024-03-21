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


import logging
import os
from datetime import datetime
from pathlib import Path

LOGGER_NAME = 'Model Compression Toolkit'


class Logger:
    # Logger has levels of verbosity.
    log_level_translate = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }
    LOG_PATH = None

    @staticmethod
    def __check_path_create_dir(log_path: str):
        """
        Create a path if not exist. Otherwise, do nothing.
        Args:
            log_path: Path to create or verify that exists.

        """

        if not os.path.exists(log_path):
            Path(log_path).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def set_logger_level(log_level=logging.INFO):
        """
        Set log level to determine the logger verbosity.
        Args:
            log_level: Level of verbosity to set for the logger.

        """

        logger = Logger.get_logger()
        logger.setLevel(log_level)

    @staticmethod
    def get_logger():
        """
        Returns: An instance of the logger.
        """
        return logging.getLogger(LOGGER_NAME)

    @staticmethod
    def set_log_file(log_folder: str = None):
        """
        Setting the logger log file path. The method gets the folder for the log file.
        In that folder, it creates a log file according to the timestamp.
        Args:
            log_folder: Folder path to hold the log file.

        """

        logger = Logger.get_logger()

        ts = datetime.now(tz=None).strftime("%d%m%Y_%H%M%S")

        if log_folder is None:
            Logger.LOG_PATH = os.path.join(os.environ.get('LOG_PATH', os.getcwd()), f"logs_{ts}")
        else:
            Logger.LOG_PATH = os.path.join(log_folder, f"logs_{ts}")
        log_name = os.path.join(Logger.LOG_PATH, f'mct_log.log')

        Logger.__check_path_create_dir(Logger.LOG_PATH)

        fh = logging.FileHandler(log_name)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)

        print(f'log file is in {log_name}')

    @staticmethod
    def shutdown():
        """
        An orderly command to shutdown by flushing and closing all logging handlers.

        """
        Logger.LOG_PATH = None
        logging.shutdown()

    ########################################
    # Delegating methods to wrapped logger
    ########################################

    @staticmethod
    def critical(msg: str):
        """
        Log a message at 'critical' severity and raise an exception.
        Args:
            msg: Message to log.

        """
        Logger.get_logger().critical(msg)
        raise Exception(msg)

    @staticmethod
    def debug(msg: str):
        """
        Log a message at 'debug' severity.

        Args:
            msg: Message to log.

        """
        Logger.get_logger().debug(msg)

    @staticmethod
    def info(msg: str):
        """
        Log a message at 'info' severity.

        Args:
            msg: Message to log.

        """
        print(msg)
        Logger.get_logger().info(msg)

    @staticmethod
    def warning(msg: str):
        """
        Log a message at 'warning' severity.

        Args:
            msg: Message to log.

        """
        print(msg)
        Logger.get_logger().warning(msg)

    @staticmethod
    def error(msg: str):
        """
        Log a message at 'error' severity and raise an exception.

        Args:
            msg: Message to log.

        """
        Logger.get_logger().error(msg)


def set_log_folder(folder: str, level: int = logging.INFO):
    """
    Set a directory path for saving a log file.

    Args:
        folder: Folder path to save the log file.
        level: Level of verbosity to set to the logger.

    """
    Logger.set_log_file(folder)
    Logger.set_logger_level(level)

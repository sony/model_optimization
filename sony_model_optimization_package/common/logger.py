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


import logging
import os
from datetime import datetime
from os import path
from pathlib import Path

LOGGER_NAME = 'Constrained Model Optimization'


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

        if not path.exists(log_path):
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
        log_name = os.path.join(Logger.LOG_PATH, f'smop_log.log')

        Logger.__check_path_create_dir(Logger.LOG_PATH)

        fh = logging.FileHandler(log_name)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)

        print(f'log file is in {log_name}')

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
    def exception(msg: str):
        """
        Log a message at 'exception' severity and raise an exception.
        Args:
            msg: Message to log.

        """
        Logger.get_logger().exception(msg)
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
        Logger.get_logger().info(msg)

    @staticmethod
    def warning(msg: str):
        """
        Log a message at 'warning' severity.

        Args:
            msg: Message to log.

        """
        Logger.get_logger().warning(msg)

    @staticmethod
    def error(msg: str):
        """
        Log a message at 'error' severity and raise an exception.

        Args:
            msg: Message to log.

        """
        Logger.get_logger().error(msg)
        raise Exception(msg)


def set_log_folder(folder: str, level: int = logging.INFO):
    """
    Set a directory path for saving a log file.

    Args:
        folder: Folder path to save the log file.
        level: Level of verbosity to set to the logger.

    """
    Logger.set_log_file(folder)
    Logger.set_logger_level(level)

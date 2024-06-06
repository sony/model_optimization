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

import json
import os
from typing import Any, Dict

import logging

from xquant.common.constants import REPORT_FILENAME
from xquant.common.tensorboard_utils import TensorboardUtils
from xquant.logger import Logger


class FrameworkReportUtils:

    def __init__(self,
                 fw_info,
                 fw_impl,
                 similarity_calculator,
                 dataset_utils,
                 model_folding,
                 tb_utils: TensorboardUtils):
        self.fw_info = fw_info
        self.fw_impl = fw_impl
        self.similarity_calculator = similarity_calculator
        self.dataset_utils = dataset_utils
        self.model_folding = model_folding
        self.tb_utils = tb_utils

    def create_report_directory(self, dir_path: str):
        """
        Create a directory for saving reports.

        Args:
            dir_path (str): The path to the directory to create.

        Returns:
            None
        """
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            Logger.get_logger().info(f"Directory created at: {dir_path}")

    def dump_report_to_json(self,
                            report_dir: str,
                            collected_data: Dict[str, Any]):
        """
        Dump the collected data into a JSON report.

        Args:
            report_dir (str): Directory where the report will be saved.
            collected_data (Dict[str, Any]): Data collected during quantization.

        Returns:
            None
        """
        report_file_name = os.path.join(report_dir, REPORT_FILENAME)
        report_file_name = os.path.abspath(report_file_name)
        Logger.get_logger().info(f"Dumping report data to: {report_file_name}")

        with open(report_file_name, 'w') as f:
            json.dump(collected_data, f, indent=4)

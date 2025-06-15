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

from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from typing import Any, Dict, Callable

from model_compression_toolkit.xquant.common.constants import REPORT_FILENAME
from model_compression_toolkit.xquant.common.dataset_utils import DatasetUtils
from model_compression_toolkit.xquant.common.model_folding_utils import ModelFoldingUtils
from model_compression_toolkit.xquant.common.similarity_calculator import SimilarityCalculator
from model_compression_toolkit.xquant.common.tensorboard_utils import TensorboardUtils
from model_compression_toolkit.logger import Logger


class FrameworkReportUtils:
    """
    Class with various utility components required for generating the report in a specific framework.
    """

    def __init__(self,
                 fw_impl: FrameworkImplementation,
                 similarity_calculator: SimilarityCalculator,
                 dataset_utils: DatasetUtils,
                 model_folding_utils: ModelFoldingUtils,
                 tb_utils: TensorboardUtils,
                 get_metadata_fn: Callable):
        """
        Initializes the FrameworkReportUtils class with various utility components required for generating the report.

        Args:
            fw_impl (FrameworkImplementation): The implemented functions of the framework.
            similarity_calculator (SimilarityCalculator): A utility for calculating similarity metrics.
            dataset_utils (DatasetUtils): Utilities for handling datasets.
            model_folding_utils (ModelFoldingUtils): Utilities for model folding operations.
            tb_utils (TensorboardUtils): Utilities for TensorBoard operations.
            get_metadata_fn (Callable): Function to retrieve the metadata from the quantized model.
        """
        self.fw_impl = fw_impl
        self.similarity_calculator = similarity_calculator
        self.dataset_utils = dataset_utils
        self.model_folding_utils = model_folding_utils
        self.tb_utils = tb_utils
        self.get_metadata_fn = get_metadata_fn

    def dump_report_to_json(self,
                            report_dir: str,
                            collected_data: Dict[str, Any]):
        """
        Dump the collected data (similarity, etc.) into a JSON file.

        Args:
            report_dir (str): Directory where the report will be saved.
            collected_data (Dict[str, Any]): Data collected during report generation.

        """
        report_file_name = os.path.join(report_dir, REPORT_FILENAME)
        report_file_name = os.path.abspath(report_file_name)
        Logger.info(f"Dumping report data to: {report_file_name}")

        with open(report_file_name, 'w') as f:
            json.dump(collected_data, f, indent=4)

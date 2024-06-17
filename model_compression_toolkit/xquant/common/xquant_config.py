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

from typing import Dict, Callable


class XQuantConfig:
    """
    Configuration for generating the report.
    It allows to set the log dir that the report will be saved in and to add similarity metrics
    to measure between tensors of the two models.
    """

    def __init__(self,
                 report_dir: str,
                 custom_similarity_metrics: Dict[str, Callable] = None):
        """
        Initializes the configuration for explainable quantization.

        Args:
            report_dir (str): Directory where the reports will be saved.
            custom_similarity_metrics (Dict[str, Callable]): Custom similarity metrics to be computed between tensors
            of the two models. The dictionary keys are similarity metric names and the values are callables that implement the
            similarity metric computation.
        """
        self.report_dir = report_dir
        self.custom_similarity_metrics = custom_similarity_metrics

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

from typing import Dict, Callable, List

from model_compression_toolkit.core.common.network_editors import EditRule


class XQuantConfig:
    def __init__(self,
                 report_dir: str,
                 compute_output_metrics_repr: bool = True,
                 compute_output_metrics_val: bool = True,
                 custom_metrics_output: Dict[str, Callable] = None,
                 compute_intermediate_metrics_repr: bool = True,
                 compute_intermediate_metrics_val: bool = False,
                 custom_metrics_intermediate: Dict[str, Callable] = None,
                 edit_rules: List[EditRule] = None):
        """
        Initializes the configuration for explainable quantization.

        Args:
            report_dir (str): Directory where the reports will be saved.
            compute_output_metrics_repr (bool): Whether to compute output metrics for the representative dataset. Default is True.
            compute_output_metrics_val (bool): Whether to compute output metrics for the validation dataset. Default is True.
            custom_metrics_output (Dict[str, Callable]): Custom metrics to be computed for the output.
                                                         The dictionary keys are metric names and the values are callables that implement the metric computation.
            compute_intermediate_metrics_repr (bool): Whether to compute intermediate metrics for the representative dataset. Default is True.
            compute_intermediate_metrics_val (bool): Whether to compute intermediate metrics for the validation dataset. Default is False.
            custom_metrics_intermediate (Dict[str, Callable]): Custom metrics to be computed for intermediate layers.
                                                                The dictionary keys are metric names and the values are callables that implement the metric computation.
            edit_rules (List[EditRule]): List of edit rules to apply before computing the metrics.
        """
        self.report_dir = report_dir
        self.compute_output_metrics_repr = compute_output_metrics_repr
        self.compute_output_metrics_val = compute_output_metrics_val
        self.custom_metrics_output = custom_metrics_output
        self.compute_intermediate_metrics_repr = compute_intermediate_metrics_repr
        self.compute_intermediate_metrics_val = compute_intermediate_metrics_val
        self.custom_metrics_intermediate = custom_metrics_intermediate
        self.edit_rules = edit_rules


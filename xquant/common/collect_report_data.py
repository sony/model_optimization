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
#
import tempfile
from typing import Callable, Any, Dict

import model_compression_toolkit as mct
from xquant import XQuantConfig
from xquant.common.constants import OUTPUT_METRICS_REPR, OUTPUT_METRICS_VAL, INTERMEDIATE_METRICS_REPR, \
    INTERMEDIATE_METRICS_VAL
from xquant.common.framework_report_utils import FrameworkReportUtils


def collect_report_data(float_model: Any,
                        quantized_model: Any,
                        repr_dataset: Callable,
                        validation_dataset: Callable,
                        core_config: mct.core.CoreConfig,
                        fw_report_utils: FrameworkReportUtils,
                        xquant_config: XQuantConfig = None) -> Dict[str, Any]:
    """
    Collects report data by computing various metrics on the quantized model.

    Args:
        float_model (Any): The original floating-point model.
        quantized_model (Any): The model after quantization.
        repr_dataset (Callable): Representative dataset used during quantization.
        validation_dataset (Callable): Validation dataset used for evaluation.
        core_config (mct.core.CoreConfig): Core configuration settings.
        fw_report_utils (FrameworkReportUtils): Utilities for generating framework-specific reports.
        xquant_config (XQuantConfig, optional): Configuration settings for explainable quantization.
                                                If not provided, a default configuration will be created.

    Returns:
        Dict[str, Any]: A dictionary containing the collected metrics and report data.
    """

    # If no xquant_config is provided, create a default one with a temporary directory for reports.
    if not xquant_config:
        report_dir = tempfile.gettempdir()
        xquant_config = XQuantConfig(report_dir=report_dir)

    fw_report_utils.create_report_directory(dir_path=xquant_config.report_dir)

    _collected_data = {}

    # Apply edit rules to the quantized model.
    if xquant_config.edit_rules:
        quantized_model = fw_report_utils.get_edited_quantized_model(float_model=float_model,
                                                                     quantized_model=quantized_model,
                                                                     xquant_config=xquant_config,
                                                                     core_config=core_config)

    # Compute output metrics for the representative dataset.
    if xquant_config.compute_output_metrics_repr:
        _collected_data[OUTPUT_METRICS_REPR] = fw_report_utils.get_metric_on_output(float_model=float_model,
                                                                                    quantized_model=quantized_model,
                                                                                    dataset=repr_dataset,
                                                                                    custom_metrics_output=xquant_config.custom_metrics_output)

    # Compute output metrics for the validation dataset.
    if xquant_config.compute_output_metrics_val:
        _collected_data[OUTPUT_METRICS_VAL] = fw_report_utils.get_metric_on_output(float_model=float_model,
                                                                                   quantized_model=quantized_model,
                                                                                   dataset=validation_dataset,
                                                                                   custom_metrics_output=xquant_config.custom_metrics_output,
                                                                                   is_validation=True)

    # Compute intermediate metrics for the representative dataset.
    if xquant_config.compute_intermediate_metrics_repr:
        _collected_data[INTERMEDIATE_METRICS_REPR] = fw_report_utils.get_metric_on_intermediate(
            float_model=float_model,
            quantized_model=quantized_model,
            dataset=repr_dataset,
            custom_metrics_intermediate=xquant_config.custom_metrics_intermediate,
            core_config=core_config)

    # Compute intermediate metrics for the validation dataset.
    if xquant_config.compute_intermediate_metrics_val:
        _collected_data[INTERMEDIATE_METRICS_VAL] = fw_report_utils.get_metric_on_intermediate(
            float_model=float_model,
            quantized_model=quantized_model,
            dataset=validation_dataset,
            custom_metrics_intermediate=xquant_config.custom_metrics_intermediate,
            core_config=core_config,
            is_validation=True)

    # Return the dictionary containing all collected data.
    return _collected_data


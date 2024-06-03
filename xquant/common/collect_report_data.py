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

from typing import Callable, Any, Dict

from xquant import XQuantConfig
from xquant.common.constants import OUTPUT_METRICS_REPR, OUTPUT_METRICS_VAL, INTERMEDIATE_METRICS_REPR, \
    INTERMEDIATE_METRICS_VAL
from xquant.common.framework_report_utils import FrameworkReportUtils


def collect_report_data(float_model: Any,
                        quantized_model: Any,
                        repr_dataset: Callable,
                        validation_dataset: Callable,
                        fw_report_utils: FrameworkReportUtils,
                        xquant_config: XQuantConfig) -> Dict[str, Any]:
    """
    Collects report data by computing various metrics on the quantized model.

    Args:
        float_model (Any): The original floating-point model.
        quantized_model (Any): The model after quantization.
        repr_dataset (Callable): Representative dataset used during quantization.
        validation_dataset (Callable): Validation dataset used for evaluation.
        fw_report_utils (FrameworkReportUtils): Utilities for generating framework-specific reports.
        xquant_config (XQuantConfig): Configuration settings for explainable quantization.

    Returns:
        Dict[str, Any]: A dictionary containing the collected metrics and report data.
    """

    fw_report_utils.create_report_directory(dir_path=xquant_config.report_dir)

    # Collect histograms and add them to Tensorboard.
    fw_report_utils.add_histograms_to_tensorboard(model=float_model,
                                                  repr_dataset=repr_dataset)

    _collected_data = {}

    # Compute output metrics for the representative dataset.
    _collected_data[OUTPUT_METRICS_REPR] = fw_report_utils.get_metric_on_output(float_model=float_model,
                                                                                quantized_model=quantized_model,
                                                                                dataset=repr_dataset,
                                                                                custom_metrics_output=xquant_config.custom_similarity_metrics)

    # Compute output metrics for the validation dataset.
    _collected_data[OUTPUT_METRICS_VAL] = fw_report_utils.get_metric_on_output(float_model=float_model,
                                                                               quantized_model=quantized_model,
                                                                               dataset=validation_dataset,
                                                                               custom_metrics_output=xquant_config.custom_similarity_metrics,
                                                                               is_validation=True)

    # Compute intermediate metrics for the representative dataset.
    _collected_data[INTERMEDIATE_METRICS_REPR] = fw_report_utils.get_metric_on_intermediate(
        float_model=float_model,
        quantized_model=quantized_model,
        dataset=repr_dataset,
        custom_metrics_intermediate=xquant_config.custom_similarity_metrics,
    )

    # Compute intermediate metrics for the validation dataset.
    _collected_data[INTERMEDIATE_METRICS_VAL] = fw_report_utils.get_metric_on_intermediate(
        float_model=float_model,
        quantized_model=quantized_model,
        dataset=validation_dataset,
        custom_metrics_intermediate=xquant_config.custom_similarity_metrics,
        is_validation=True)

    # Generate the quantized graph with metrics.
    quant_graph = fw_report_utils.get_quant_graph_with_metrics(quantized_model=quantized_model,
                                                               collected_data=_collected_data,
                                                               xquant_config=xquant_config,
                                                               repr_dataset=repr_dataset)

    # Add the quantized graph to TensorBoard for visualization.
    fw_report_utils.add_graph_to_tensorboard(graph=quant_graph)

    fw_report_utils.dump_report_to_json(report_dir=xquant_config.report_dir,
                                        collected_data=_collected_data)

    return _collected_data


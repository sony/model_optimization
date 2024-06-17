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
from tqdm import tqdm
from typing import Callable, Any, Dict

from model_compression_toolkit.core.common.model_collector import ModelCollector
from model_compression_toolkit.xquant import XQuantConfig
from model_compression_toolkit.xquant.common.constants import OUTPUT_SIMILARITY_METRICS_REPR, OUTPUT_SIMILARITY_METRICS_VAL, INTERMEDIATE_SIMILARITY_METRICS_REPR, \
    INTERMEDIATE_SIMILARITY_METRICS_VAL
from model_compression_toolkit.xquant.common.framework_report_utils import FrameworkReportUtils


def core_report_generator(float_model: Any,
                          quantized_model: Any,
                          repr_dataset: Callable,
                          validation_dataset: Callable,
                          fw_report_utils: FrameworkReportUtils,
                          xquant_config: XQuantConfig) -> Dict[str, Any]:
    """
    Generate report in tensorboard with a graph of the quantized model and similarity metrics that
    have been measured when comparing to the float model (or any other two models).
    The report also contains histograms that are collected on the baseline model (usually, the float
    model).

    Args:
        float_model (Any): The original floating-point model.
        quantized_model (Any): The model after quantization.
        repr_dataset (Callable): Representative dataset used for similarity metrics computation.
        validation_dataset (Callable): Validation dataset used for similarity metrics computation.
        fw_report_utils (FrameworkReportUtils): Utilities for generating framework-specific reports.
        xquant_config (XQuantConfig): Configuration settings for explainable quantization.

    Returns:
        Dict[str, Any]: A dictionary containing the collected similarity metrics and report data.
    """
    # Collect histograms on the float model.
    float_graph = fw_report_utils.model_folding_utils.create_float_folded_graph(float_model, repr_dataset)
    mi = ModelCollector(float_graph, fw_report_utils.fw_impl, fw_report_utils.fw_info)
    for _data in tqdm(repr_dataset(), desc="Collecting Histograms"):
        mi.infer(_data)

    # Collect histograms and add them to Tensorboard.
    fw_report_utils.tb_utils.add_histograms_to_tensorboard(graph=float_graph)

    # Compute similarity metrics on representative dataset and validation set.
    repr_similarity = fw_report_utils.similarity_calculator.compute_similarity_metrics(float_model=float_model,
                                                                                       quantized_model=quantized_model,
                                                                                       dataset=repr_dataset,
                                                                                       custom_similarity_metrics=xquant_config.custom_similarity_metrics)
    val_similarity = fw_report_utils.similarity_calculator.compute_similarity_metrics(float_model=float_model,
                                                                                      quantized_model=quantized_model,
                                                                                      dataset=validation_dataset,
                                                                                      custom_similarity_metrics=xquant_config.custom_similarity_metrics,
                                                                                      is_validation=True)
    similarity_metrics = {
        OUTPUT_SIMILARITY_METRICS_REPR: repr_similarity[0],
        OUTPUT_SIMILARITY_METRICS_VAL: val_similarity[0],
        INTERMEDIATE_SIMILARITY_METRICS_REPR: repr_similarity[1],
        INTERMEDIATE_SIMILARITY_METRICS_VAL: val_similarity[1]
    }

    # Add a graph of the quantized model with the similarity metrics to TensorBoard for visualization.
    fw_report_utils.tb_utils.add_graph_to_tensorboard(quantized_model,
                                                      similarity_metrics,
                                                      repr_dataset)

    # Save data to a json file.
    fw_report_utils.dump_report_to_json(report_dir=xquant_config.report_dir,
                                        collected_data=similarity_metrics)

    return similarity_metrics

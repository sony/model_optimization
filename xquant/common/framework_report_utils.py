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
from typing import Tuple, Any, Callable, Dict

from model_compression_toolkit.core.common import Graph
from xquant import XQuantConfig
from xquant.common.constants import CS_METRIC_NAME, SQNR_METRIC_NAME, MSE_METRIC_NAME, REPORT_FILENAME
from xquant.logger import Logger

DEFAULT_METRICS_NAMES = [CS_METRIC_NAME,
                         MSE_METRIC_NAME,
                         SQNR_METRIC_NAME]

class FrameworkReportUtils:

    def __init__(self,
                 fw_info,
                 fw_impl,
                 similarity_metrics,
                 dataset_utils,
                 model_folding,
                 tb_utils):
        self.fw_info = fw_info
        self.fw_impl = fw_impl
        self.similarity_metrics = similarity_metrics
        self.dataset_utils = dataset_utils
        self.model_folding = model_folding
        self.tb_utils = tb_utils


    def get_quantized_graph(self,
                            quantized_model: Any,
                            repr_dataset: Callable):
        """
        Get a graph representation of the quantized model.

        Args:
            quantized_model: The quantized model.
            repr_dataset: Representative dataset to use during the graph building.

        Returns:
            Graph representation of the quantized model.
        """

        Logger.critical(f"get_quantized_graph is not implemented.")

    def get_quant_graph_with_metrics(self,
                                     quantized_model: Any,
                                     collected_data: Dict[str, Any],
                                     xquant_config: XQuantConfig,
                                     repr_dataset: Callable):
        """
        Generate the quantized graph with associated metrics.

        Args:
            quantized_model (Any): The model after quantization.
            collected_data (Dict[str, Any]): Data collected during quantization.
            xquant_config (XQuantConfig): Configuration settings for explainable quantization.
            repr_dataset (Callable): Representative dataset used during graph building.

        Returns:
            Any: A graph structure with metrics.
        """
        Logger.critical(f"get_quant_graph_with_metrics is not implemented.")

    def get_metric_on_output(self,
                             float_model: Any,
                             quantized_model: Any,
                             dataset: Callable,
                             custom_similarity_metrics: Dict[str, Callable] = None,
                             is_validation: bool = False) -> Dict[str, float]:
        """
        Compute metrics on the output of the model.

        Args:
            float_model (Any): The original floating-point model.
            quantized_model (Any): The model after quantization.
            dataset (Callable): Dataset used for evaluation.
            custom_similarity_metrics (Dict[str, Callable], optional): Custom metrics for output evaluation. Defaults to None.
            is_validation (bool, optional): Flag indicating if this is a validation dataset. Defaults to False.

        Returns:
            Dict[str, float]: A dictionary of computed metrics.
        """
        Logger.critical(f"get_metric_on_output is not implemented.")

    def get_metric_on_intermediate(self,
                                   float_model: Any,
                                   quantized_model: Any,
                                   dataset: Callable,
                                   custom_similarity_metrics: Dict[str, Callable] = None,
                                   is_validation: bool = False) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics on intermediate layers of the model.

        Args:
            float_model (Any): The original floating-point model.
            quantized_model (Any): The model after quantization.
            dataset (Callable): Dataset used for evaluation.
            custom_similarity_metrics (Dict[str, Callable], optional): Custom metrics for intermediate layers. Defaults to None.
            is_validation (bool, optional): Flag indicating if this is a validation dataset. Defaults to False.

        Returns:
            Dict[str, Dict[str, float]]: A dictionary of computed metrics for intermediate layers.
        """
        Logger.critical(f"get_metric_on_intermediate is not implemented.")



    def get_float_to_quantized_compare_points(self,
                                              quantized_model: Any,
                                              float_model: Any) -> Dict[str, str]:
        """
        Get comparison points between the floating-point and quantized models.

        Args:
            quantized_model (Any): The quantized model.
            float_model (Any): The floating-point model.

        Returns:
            Dict[str, str]: A dictionary mapping comparison points between the two models.
        """
        Logger.critical(f"get_float_to_quantized_compare_points is not implemented.")

    def compute_metrics(self,
                        predictions: Tuple[Any, Any],
                        similarity_metrics: Dict[str, Callable]) -> Dict[str, float]:
        """
        Compute metrics based on predictions.

        Args:
            predictions (Tuple[Any, Any]): A tuple of predictions from the floating-point and quantized models.
            similarity_metrics (Dict[str, Callable]): Custom metrics for output evaluation.

        Returns:
            Dict[str, float]: A dictionary of computed metrics.
        """

        f_pred, q_pred = predictions

        # Compute default metrics
        metrics = {k: v(f_pred, q_pred) for k, v in similarity_metrics.items()}

        return metrics


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
            Logger.info(f"Directory created at: {dir_path}")

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
        Logger.info(f"Dumping report data to: {report_file_name}")

        with open(report_file_name, 'w') as f:
            json.dump(collected_data, f, indent=4)

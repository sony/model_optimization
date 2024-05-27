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
#
import json
import os
from typing import Tuple, Any, Callable, Dict

import model_compression_toolkit as mct
from model_compression_toolkit.core import FrameworkInfo
from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.visualization.tensorboard_writer import TensorboardWriter
from xquant import XQuantConfig
from xquant.common.constants import CS_METRIC_NAME, SQNR_METRIC_NAME, MSE_METRIC_NAME, REPORT_FILENAME

DEFAULT_METRICS_NAMES = [CS_METRIC_NAME, MSE_METRIC_NAME, SQNR_METRIC_NAME]

class FrameworkReportUtils:

    def get_quant_graph_with_metrics(self,
                                     quantized_model: Any,
                                     collected_data: Dict[str, Any],
                                     xquant_config: XQuantConfig):
        """
        Generate the quantized graph with associated metrics.

        Args:
            quantized_model (Any): The model after quantization.
            collected_data (Dict[str, Any]): Data collected during quantization.
            xquant_config (XQuantConfig): Configuration settings for explainable quantization.

        Returns:
            Any: A graph structure with metrics.
        """
        raise NotImplemented

    def get_edited_quantized_model(self,
                                   float_model: Any,
                                   quantized_model: Any,
                                   xquant_config: XQuantConfig,
                                   core_config: mct.core.CoreConfig) -> Any:
        """
        Edit the quantized model based on the given configuration.

        Args:
            float_model (Any): The original floating-point model.
            quantized_model (Any): The model after quantization.
            xquant_config (XQuantConfig): Configuration settings for explainable quantization.
            core_config (mct.core.CoreConfig): Core configuration settings.

        Returns:
            Any: The edited quantized model.
        """
        raise NotImplemented

    def get_metric_on_output(self,
                             float_model: Any,
                             quantized_model: Any,
                             dataset: Callable,
                             custom_metrics_output: Dict[str, Callable] = None,
                             is_validation: bool = False) -> Dict[str, float]:
        """
        Compute metrics on the output of the model.

        Args:
            float_model (Any): The original floating-point model.
            quantized_model (Any): The model after quantization.
            dataset (Callable): Dataset used for evaluation.
            custom_metrics_output (Dict[str, Callable], optional): Custom metrics for output evaluation. Defaults to None.
            is_validation (bool, optional): Flag indicating if this is a validation dataset. Defaults to False.

        Returns:
            Dict[str, float]: A dictionary of computed metrics.
        """
        raise NotImplemented

    def get_metric_on_intermediate(self,
                                   float_model: Any,
                                   quantized_model: Any,
                                   dataset: Callable,
                                   core_config: mct.core.CoreConfig,
                                   custom_metrics_intermediate: Dict[str, Callable] = None,
                                   is_validation: bool = False) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics on intermediate layers of the model.

        Args:
            float_model (Any): The original floating-point model.
            quantized_model (Any): The model after quantization.
            dataset (Callable): Dataset used for evaluation.
            core_config (mct.core.CoreConfig): Core configuration settings.
            custom_metrics_intermediate (Dict[str, Callable], optional): Custom metrics for intermediate layers. Defaults to None.
            is_validation (bool, optional): Flag indicating if this is a validation dataset. Defaults to False.

        Returns:
            Dict[str, Dict[str, float]]: A dictionary of computed metrics for intermediate layers.
        """
        raise NotImplemented

    def get_default_metrics(self) -> Dict[str, Callable]:
        """
        Retrieve the default metrics for evaluation.

        Returns:
            Dict[str, Callable]: A dictionary of default metric functions.
        """
        raise NotImplemented

    def create_float_folded_model(self,
                                  float_model: Any,
                                  representative_dataset: Any,
                                  core_config: mct.core.CoreConfig):
        """
        Create a folded version of the floating-point model.

        Args:
            float_model (Any): The floating-point model.
            representative_dataset (Any): Representative dataset used during folding.
            core_config (mct.core.CoreConfig): Core configuration settings.

        Returns:
            Any: The folded floating-point model.
        """
        raise NotImplemented

    def wrapped_dataset(self,
                        dataset: Any,
                        is_validation: bool,
                        device: str):
        """
        Wrap the dataset for handling labeled and unlabeled datasets.

        Args:
            dataset (Any): The dataset to wrap.
            is_validation (bool): Flag indicating if this is a validation dataset.
            device (str): Device to use for evaluation.

        Returns:
            Any: Wrapped dataset ready for evaluation.
        """
        raise NotImplemented

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
        raise NotImplemented

    def compute_metrics(self,
                        predictions: Tuple[Any, Any],
                        custom_metrics_output: Dict[str, Callable] = None) -> Dict[str, float]:
        """
        Compute metrics based on predictions.

        Args:
            predictions (Tuple[Any, Any]): A tuple of predictions from the floating-point and quantized models.
            custom_metrics_output (Dict[str, Callable], optional): Custom metrics for output evaluation. Defaults to None.

        Returns:
            Dict[str, float]: A dictionary of computed metrics.
        """

        f_pred, q_pred = predictions

        # Compute default metrics
        metrics = {k: v(f_pred=f_pred, q_pred=q_pred) for k, v in self.get_default_metrics().items()}

        # Compute custom metrics if provided
        if custom_metrics_output:
            for metric_name, metric_fn in custom_metrics_output.items():
                metrics[metric_name] = metric_fn(f_pred, q_pred)

        return metrics

    def add_graph_to_tensorboard(self,
                                 graph: Graph,
                                 fw_info: FrameworkInfo,
                                 report_dir: str):
        """
        Add the graph to TensorBoard for visualization.

        Args:
            graph (Graph): The graph to add.
            fw_info (FrameworkInfo): Framework information.
            report_dir (str): Directory where the TensorBoard logs will be saved.

        Returns:
            None
        """
        tb_writer = TensorboardWriter(report_dir, fw_info)
        tb_writer.add_graph(graph, "")
        tb_writer.close()
        print(f"Please run: tensorboard --logdir {report_dir}")

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
            print(f"Directory created at: {dir_path}")

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
        print(f"Dumping report data to: {report_file_name}")

        with open(report_file_name, 'w') as f:
            json.dump(collected_data, f, indent=4)

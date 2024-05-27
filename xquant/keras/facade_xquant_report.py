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
from typing import Callable, Dict, Any

import model_compression_toolkit as mct
import keras

from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from xquant.common.collect_report_data import collect_report_data
from xquant import XQuantConfig
from xquant.common.constants import FOUND_TF
from xquant.keras.keras_report_utils import KerasReportUtils
from xquant.logger import Logger

if FOUND_TF:
    def xquant_report_keras_experimental(float_model: keras.Model,
                                         quantized_model: keras.Model,
                                         repr_dataset: Callable,
                                         validation_dataset: Callable,
                                         core_config: mct.core.CoreConfig,
                                         xquant_config: XQuantConfig = None) -> Dict[str, Any]:
        """
        Generate an explainable quantization report for Keras models.

        Args:
            float_model (keras.Model): The original floating-point Keras model.
            quantized_model (keras.Model): The quantized Keras model.
            repr_dataset (Callable): The representative dataset used for evaluation.
            validation_dataset (Callable): The validation dataset used for evaluation.
            core_config (mct.core.CoreConfig): Core configuration settings for the model.
            xquant_config (XQuantConfig, optional): Configuration settings for explainable quantization.
                                                    If not provided, a default configuration will be created.

        Returns:
            Dict[str, Any]: A dictionary containing the collected metrics and report data.
        """

        # Initialize a utility class for handling Keras-specific reporting tasks.
        keras_report_utils = KerasReportUtils()

        # Collect data and metrics for the report.
        _collected_data = collect_report_data(float_model,
                                              quantized_model,
                                              repr_dataset,
                                              validation_dataset,
                                              core_config,
                                              keras_report_utils,
                                              xquant_config)

        # Generate the quantized graph with metrics.
        quant_graph = keras_report_utils.get_quant_graph_with_metrics(quantized_model=quantized_model,
                                                                      collected_data=_collected_data,
                                                                      xquant_config=xquant_config)

        # Add the quantized graph to TensorBoard for visualization.
        keras_report_utils.add_graph_to_tensorboard(graph=quant_graph,
                                                    fw_info=DEFAULT_KERAS_INFO,
                                                    report_dir=xquant_config.report_dir)

        keras_report_utils.dump_report_to_json(report_dir=xquant_config.report_dir,
                                               collected_data=_collected_data)

        return _collected_data
else:
    def xquant_report_keras_experimental(*args, **kwargs):
        Logger.critical("Tensorflow must be installed to use xquant_report_keras_experimental. "
                        "The 'tensorflow' package is missing.")  # pragma: no cover


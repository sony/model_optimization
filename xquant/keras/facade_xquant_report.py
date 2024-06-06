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
import logging

from typing import Callable, Dict, Any


from xquant.common.core_report_generator import core_report_generator
from xquant import XQuantConfig
from xquant.common.constants import FOUND_TF
from xquant.logger import Logger

if FOUND_TF:
    import keras
    from xquant.keras.keras_report_utils import KerasReportUtils

    def xquant_report_keras_experimental(float_model: keras.Model,
                                         quantized_model: keras.Model,
                                         repr_dataset: Callable,
                                         validation_dataset: Callable,
                                         xquant_config: XQuantConfig) -> Dict[str, Any]:
        """
        Generate an explainable quantization report for a quantized Keras model.

        Args:
            float_model (keras.Model): The original floating-point Keras model.
            quantized_model (keras.Model): The quantized Keras model.
            repr_dataset (Callable): The representative dataset used for evaluation.
            validation_dataset (Callable): The validation dataset used for evaluation.
            xquant_config (XQuantConfig, optional): Configuration settings for explainable quantization.

        Returns:
            Dict[str, Any]: A dictionary containing the collected metrics and report data.
        """
        # Initialize the logger with the report directory.
        Logger.get_logger(log_dir=xquant_config.report_dir)

        # Initialize a utility class for handling Keras-specific reporting tasks.
        keras_report_utils = KerasReportUtils(xquant_config.report_dir)

        # Collect data and metrics for the report.
        _collected_data = core_report_generator(float_model=float_model,
                                                quantized_model=quantized_model,
                                                repr_dataset=repr_dataset,
                                                validation_dataset=validation_dataset,
                                                fw_report_utils=keras_report_utils,
                                                xquant_config=xquant_config)

        return _collected_data
else:
    def xquant_report_keras_experimental(*args, **kwargs):
        Logger.get_logger().critical("Tensorflow must be installed to use xquant_report_keras_experimental. "
                        "The 'tensorflow' package is missing.")  # pragma: no cover


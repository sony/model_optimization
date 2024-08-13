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

from typing import Callable, Dict, Any

from model_compression_toolkit.verify_packages import FOUND_TF
from model_compression_toolkit.xquant.common.core_report_generator import core_report_generator
from model_compression_toolkit.xquant import XQuantConfig
from model_compression_toolkit.logger import Logger

if FOUND_TF:
    import keras
    from model_compression_toolkit.xquant.keras.keras_report_utils import KerasReportUtils

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
            repr_dataset (Callable): The representative dataset used during quantization for similarity metrics computation.
            validation_dataset (Callable): The validation dataset used for evaluation for similarity metrics computation.
            xquant_config (XQuantConfig): Configuration settings for explainable quantization.

        Returns:
            Dict[str, Any]: A dictionary containing the collected similarity metrics and report data.
        """
        # Initialize the logger with the report directory.
        Logger.set_log_file(log_folder=xquant_config.report_dir)

        # Initialize a utility class for handling Keras-specific reporting tasks.
        keras_report_utils = KerasReportUtils(xquant_config.report_dir)

        # Create the report after collecting useful data like histograms and similarity metrics.
        _collected_data = core_report_generator(float_model=float_model,
                                                quantized_model=quantized_model,
                                                repr_dataset=repr_dataset,
                                                validation_dataset=validation_dataset,
                                                fw_report_utils=keras_report_utils,
                                                xquant_config=xquant_config)

        Logger.shutdown()

        return _collected_data
else:
    def xquant_report_keras_experimental(*args, **kwargs):
        Logger.critical("Tensorflow must be installed with a version of 2.15 or lower to use "
                        "xquant_report_keras_experimental. The 'tensorflow' package is missing "
                        "or is installed with a version higher than 2.15.")  # pragma: no cover

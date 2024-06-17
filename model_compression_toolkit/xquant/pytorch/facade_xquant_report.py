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

from typing import Callable

from model_compression_toolkit.constants import FOUND_TORCH
from model_compression_toolkit.xquant.common.core_report_generator import core_report_generator
from model_compression_toolkit.xquant import XQuantConfig
from model_compression_toolkit.logger import Logger

if FOUND_TORCH:
    from model_compression_toolkit.xquant.pytorch.pytorch_report_utils import PytorchReportUtils
    import torch

    def xquant_report_pytorch_experimental(float_model: torch.nn.Module,
                                           quantized_model: torch.nn.Module,
                                           repr_dataset: Callable,
                                           validation_dataset: Callable,
                                           xquant_config: XQuantConfig):
        """
        Generate an explainable quantization report for a quantized Pytorch model.

        Args:
            float_model (torch.nn.Module): The original floating-point Pytorch model.
            quantized_model (torch.nn.Module): The quantized Pytorch model.
            repr_dataset (Callable): The representative dataset used during quantization.
            validation_dataset (Callable): The validation dataset used for evaluation.
            xquant_config (XQuantConfig): Configuration settings for explainable quantization.

        Returns:
            Dict[str, Any]: A dictionary containing the collected similarity metrics and report data.
        """
        # Initialize the logger with the report directory.
        Logger.set_log_file(log_folder=xquant_config.report_dir)

        pytorch_report_utils = PytorchReportUtils(xquant_config.report_dir)

        _collected_data = core_report_generator(float_model=float_model,
                                                quantized_model=quantized_model,
                                                repr_dataset=repr_dataset,
                                                validation_dataset=validation_dataset,
                                                fw_report_utils=pytorch_report_utils,
                                                xquant_config=xquant_config)

        return _collected_data

else:
    def xquant_report_pytorch_experimental(*args, **kwargs):
        Logger.critical("PyTorch must be installed to use 'xquant_report_pytorch_experimental'. "
                                     "The 'torch' package is missing.")  # pragma: no cover

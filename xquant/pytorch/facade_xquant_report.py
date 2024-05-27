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

from typing import Callable
import model_compression_toolkit as mct
import torch

from model_compression_toolkit.core.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO
from xquant.common.collect_report_data import collect_report_data
from xquant import XQuantConfig

from xquant.pytorch.pytorch_report_utils import PytorchReportUtils


def xquant_report_pytorch_experimental(float_model: torch.nn.Module,
                                       quantized_model: torch.nn.Module,
                                       repr_dataset: Callable,
                                       validation_dataset: Callable,
                                       core_config: mct.core.CoreConfig,
                                       xquant_config: XQuantConfig = None):

    pytorch_report_utils = PytorchReportUtils()
    _collected_data = collect_report_data(float_model,
                                          quantized_model,
                                          repr_dataset,
                                          validation_dataset,
                                          core_config,
                                          pytorch_report_utils,
                                          xquant_config)

    quant_graph = pytorch_report_utils.get_quant_graph_with_metrics(quantized_model=quantized_model,
                                                                    collected_data=_collected_data,
                                                                    xquant_config=xquant_config)
    pytorch_report_utils.add_graph_to_tensorboard(graph=quant_graph,
                                                  fw_info=DEFAULT_PYTORCH_INFO,
                                                  report_dir=xquant_config.report_dir)
    pytorch_report_utils.dump_report_to_json(report_dir=xquant_config.report_dir,
                                             collected_data=_collected_data)

    return _collected_data

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
from typing import Dict, Any, Callable

import torch

from model_compression_toolkit.core.pytorch.reader.reader import model_reader
from xquant import XQuantConfig
from xquant.common.constants import XQUANT_REPR, INTERMEDIATE_METRICS_REPR, XQUANT_VAL, INTERMEDIATE_METRICS_VAL
from xquant.common.model_folding_utils import ModelFoldingUtils
from xquant.common.tensorboard_utils import TensorboardUtils


class PytorchTensorboardUtils(TensorboardUtils):

    def __init__(self, report_dir: str, model_folding_utils: ModelFoldingUtils, fw_info, fw_impl):
        super().__init__(report_dir, model_folding_utils, fw_info, fw_impl)

    def get_graph_for_tensorboard_display(self,
                                          quantized_model: torch.nn.Module,
                                          similarity_metrics: Dict[str, Any],
                                          xquant_config: XQuantConfig,
                                          repr_dataset: Callable):
        """
        Updates the quantized model graph with metrics data collected during evaluation.

        Args:
            quantized_model: The quantized model.
            similarity_metrics: Dictionary containing the collected metrics data.
            xquant_config: Configuration settings for quantization.
            repr_dataset: Representative dataset used during graph building.

        Returns:
            The updated quantized model graph.
        """
        quant_graph = model_reader(quantized_model,
                                   representative_data_gen=repr_dataset,
                                   to_tensor=self.fw_impl.to_tensor,
                                   to_numpy=self.fw_impl.to_numpy)

        for node in quant_graph.nodes:
            if node.name in similarity_metrics[INTERMEDIATE_METRICS_REPR].keys():
                node.framework_attr[XQUANT_REPR] = similarity_metrics[INTERMEDIATE_METRICS_REPR][f"{node.name}"]
            elif node.name.removesuffix("_layer") in similarity_metrics[INTERMEDIATE_METRICS_REPR].keys():
                node.framework_attr[XQUANT_REPR] = similarity_metrics[INTERMEDIATE_METRICS_REPR][
                    node.name.removesuffix("_layer")]

            if node.name in similarity_metrics[INTERMEDIATE_METRICS_VAL].keys():
                node.framework_attr[XQUANT_VAL] = similarity_metrics[INTERMEDIATE_METRICS_VAL][f"{node.name}"]
            elif node.name.removesuffix("_layer") in similarity_metrics[INTERMEDIATE_METRICS_VAL].keys():
                node.framework_attr[XQUANT_VAL] = similarity_metrics[INTERMEDIATE_METRICS_VAL][
                    node.name.removesuffix("_layer")]
        return quant_graph

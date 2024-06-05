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
from model_compression_toolkit.core.keras.reader.reader import model_reader

from xquant.common.constants import XQUANT_REPR, INTERMEDIATE_METRICS_REPR, XQUANT_VAL, INTERMEDIATE_METRICS_VAL
from xquant.common.model_folding_utils import ModelFoldingUtils
from xquant.common.tensorboard_utils import TensorboardUtils


class KerasTensorboardUtils(TensorboardUtils):

    def __init__(self, report_dir: str, model_folding_utils: ModelFoldingUtils, fw_info, fw_impl):
        super().__init__(report_dir, model_folding_utils, fw_info, fw_impl)
    def get_graph_for_tensorboard_display(self,
                                          quantized_model,
                                          similarity_metrics,
                                          xquant_config,
                                          repr_dataset):
        quant_graph = model_reader(quantized_model)
        for node in quant_graph.nodes:
            if node.name in similarity_metrics[INTERMEDIATE_METRICS_REPR].keys():
                node.framework_attr[XQUANT_REPR] = similarity_metrics[INTERMEDIATE_METRICS_REPR][node.name]
            if node.name in similarity_metrics[INTERMEDIATE_METRICS_VAL].keys():
                node.framework_attr[XQUANT_VAL] = similarity_metrics[INTERMEDIATE_METRICS_VAL][node.name]
        return quant_graph

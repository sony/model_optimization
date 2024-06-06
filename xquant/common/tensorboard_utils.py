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

from tqdm import tqdm
from typing import Callable

from model_compression_toolkit.core.common.model_collector import ModelCollector
from model_compression_toolkit.core.common.visualization.tensorboard_writer import TensorboardWriter
from xquant.common.model_folding_utils import ModelFoldingUtils
from xquant.logger import Logger


class TensorboardUtils:

    def __init__(self, report_dir: str, model_folding_utils: ModelFoldingUtils, fw_info, fw_impl):
        self.fw_impl = fw_impl
        self.fw_info = fw_info
        self.model_folding_utils = model_folding_utils
        self.tb_writer = TensorboardWriter(report_dir, fw_info)
        Logger.get_logger().info(f"Please run: tensorboard --logdir {self.tb_writer.dir_path}")

    def add_histograms_to_tensorboard(self, model, repr_dataset: Callable):
        graph = self.model_folding_utils.create_float_folded_graph(model, repr_dataset)
        mi = ModelCollector(graph, self.fw_impl, self.fw_info)
        for _data in tqdm(repr_dataset(), "Collecting Histograms"):
            mi.infer(_data)
        self.tb_writer.add_histograms(graph, "")
    def get_graph_for_tensorboard_display(self,
                                          quantized_model,
                                          similarity_metrics,
                                          xquant_config,
                                          repr_dataset):
        raise NotImplemented
    def add_graph_to_tensorboard(self,
                                 quantized_model,
                                 similarity_metrics,
                                 xquant_config,
                                 repr_dataset
                                 ):
        # Generate the quantized graph with metrics.
        tb_graph = self.get_graph_for_tensorboard_display(quantized_model=quantized_model,
                                                          similarity_metrics=similarity_metrics,
                                                          xquant_config=xquant_config,
                                                          repr_dataset=repr_dataset)
        self.tb_writer.add_graph(tb_graph, "")



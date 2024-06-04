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

from model_compression_toolkit.core.common.model_builder_mode import ModelBuilderMode
from model_compression_toolkit.core.common.quantization.quantization_config import DEFAULTCONFIG

from model_compression_toolkit.core.graph_prep_runner import graph_preparation_runner
from typing import Any, Callable

from model_compression_toolkit.core.common import Graph


class ModelFoldingUtils:

    def __init__(self, fw_info, fw_impl, fw_default_tpc):
        self.fw_info = fw_info
        self.fw_impl = fw_impl
        self.fw_default_tpc = fw_default_tpc

    def create_float_folded_model(self, float_model: Any, representative_dataset: Any=None):
        float_graph = self.create_float_folded_graph(model=float_model, repr_dataset=representative_dataset)
        float_folded_model, _ = self.fw_impl.model_builder(
            float_graph,
            mode=ModelBuilderMode.FLOAT,
            append2output=None,
            fw_info=self.fw_info
        )
        return float_folded_model

    def create_float_folded_graph(self, model: Any, repr_dataset: Callable) -> Graph:
        graph = graph_preparation_runner(in_model=model,
                                         representative_data_gen=repr_dataset,
                                         fw_impl=self.fw_impl,
                                         fw_info=self.fw_info,
                                         quantization_config=DEFAULTCONFIG,
                                         tpc=self.fw_default_tpc)
        return graph


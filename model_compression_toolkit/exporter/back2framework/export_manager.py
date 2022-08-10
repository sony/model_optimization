# Copyright 2022 Sony Semiconductors Israel, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


# from model_compression_toolkit.core.common import FrameworkInfo
# from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
# from model_compression_toolkit.core.common.graph.base_graph import Graph
# from model_compression_toolkit.core.common.model_builder_mode import ModelBuilderMode
# from model_compression_toolkit.core.common.quantization.quantize_graph_weights import quantize_graph_weights
# from model_compression_toolkit.core.common.substitutions.apply_substitutions import substitute
# from model_compression_toolkit.core.common.user_info import UserInformation
# from model_compression_toolkit.core.common.visualization.tensorboard_writer import TensorboardWriter
from model_compression_toolkit.core.common.back2framework.base_model_builder import BaseModelBuilder
from model_compression_toolkit.exporter.back2framework.base_exporter import BaseExporter


class ExporterManager:

    def __init__(self,
                 exporter: BaseExporter,
                 ):
        self.exporter = exporter

    def export(self):
        complete_info_model = self.exporter.build_model()
        self._validate_model(complete_info_model)
        return complete_info_model

    def _validate_model(self, complete_info_model):
        raise NotImplemented



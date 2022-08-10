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


from typing import Tuple, Any, List, Callable

# from model_compression_toolkit.core.common import FrameworkInfo
# from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
# from model_compression_toolkit.core.common.graph.base_graph import Graph
# from model_compression_toolkit.core.common.model_builder_mode import ModelBuilderMode
# from model_compression_toolkit.core.common.quantization.quantize_graph_weights import quantize_graph_weights
# from model_compression_toolkit.core.common.substitutions.apply_substitutions import substitute
# from model_compression_toolkit.core.common.user_info import UserInformation
# from model_compression_toolkit.core.common.visualization.tensorboard_writer import TensorboardWriter
from model_compression_toolkit.exporter.back2framework.base_model_builder import BaseModelBuilder


class ExporterManager:

    def __init__(self,
                 model_builder: BaseModelBuilder,
                 ):
        self.model_builder = model_builder

    def export(self):
        complete_info_model = self.model_builder.build_model()
        self._validate_model(complete_info_model)
        return complete_info_model

    def _validate_model(self, complete_info_model):
        pass


#
#
# def experimental_export_model(tg: Graph,
#                               fw_info: FrameworkInfo,
#                               fw_impl: FrameworkImplementation,
#                               tb_w: TensorboardWriter,
#                               bit_widths_config: List[int]):
#     """
#     A function for quantizing the graph's weights and build a quantized framework model from it.
#
#     Args:
#         tg: A prepared for quantization graph.
#         fw_info: Information needed for quantization about the specific framework (e.g., kernel channels indices,
#         groups of layers by how they should be quantized, etc.).
#         fw_impl: FrameworkImplementation object with a specific framework methods implementation.
#         tb_w: TensorBoardWriter object to log events.
#         bit_widths_config: mixed-precision bit configuration to be added to model user_info
#
#     Returns:
#         Quantized model in the input framework, and information the user may need in order to use the quantized model.
#     """
#
#     # quantized_tg = substitute(tg, fw_impl.get_substitutions_pre_build())
#
#
#     quantized_model, user_info = fw_impl.model_builder(quantized_tg,
#                                                        mode=ModelBuilderMode.FULLY_QUANTIZED,
#                                                        fw_info=fw_info)
#     user_info.mixed_precision_cfg = bit_widths_config
#
#     return quantized_model, user_info

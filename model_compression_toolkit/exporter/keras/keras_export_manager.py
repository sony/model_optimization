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

from model_compression_toolkit.exporter.back2framework.base_model_builder import BaseModelBuilder
from model_compression_toolkit.exporter.back2framework.export_manager import ExporterManager
from model_compression_toolkit.exporter.keras.complete_info_validate import validate_complete_quantization_info


class KerasExporterManager(ExporterManager):

    def __init__(self,
                 model_builder: BaseModelBuilder,
                 ):
        super(KerasExporterManager, self).__init__(model_builder=model_builder)
        self.validate_model_fn = validate_complete_quantization_info

    def export(self):
        complete_info_model, user_info = self.model_builder.build_model()
        self.validate_model_fn(complete_info_model)
        return complete_info_model, user_info



# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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
from typing import Callable

import torch.nn

from model_compression_toolkit.exporter.model_exporter.fw_agonstic.exporter import Exporter


class BasePyTorchExporter(Exporter):
    """
    Base PyTorch exporter class.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 is_layer_exportable_fn: Callable,
                 save_model_path: str,
                 repr_dataset: Callable):
        """
        Args:
            model: Model to export.
            is_layer_exportable_fn: Callable to check whether a layer can be exported or not.
            save_model_path: Path to save the exported model.
            repr_dataset: Representative dataset (needed for creating torch script).

        """
        super().__init__(model,
                         is_layer_exportable_fn,
                         save_model_path)
        self.repr_dataset = repr_dataset

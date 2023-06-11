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

from model_compression_toolkit.logger import Logger
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor
from model_compression_toolkit.exporter.model_exporter.pytorch.base_pytorch_exporter import BasePyTorchExporter


class FakelyQuantTorchScriptPyTorchExporter(BasePyTorchExporter):
    """
    Exporter for fakely-quant PyTorch models.
    The exporter expects to receive an exportable model (where each layer's full quantization parameters
    can be retrieved), and convert it into a fakely-quant model (namely, weights that are in fake-quant
    format) and fake-quant layers for the activations.
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
                         save_model_path,
                         repr_dataset)

    def export(self) -> None:
        """
        Convert an exportable (fully-quantized) PyTorch model to a fakely-quant model
        (namely, weights that are in fake-quant format) and fake-quant layers for the activations.

        Returns:
            Fake-quant PyTorch model.
        """
        for layer in self.model.children():
            self.is_layer_exportable_fn(layer)

        self._substitute_fully_quantized_model()

        torch_traced = torch.jit.trace(self.model,
                                       to_torch_tensor(next(self.repr_dataset())),
                                       check_trace=True)

        self.exported_model = torch.jit.script(torch_traced)

        Logger.info(f"Exporting PyTorch torch script Model: {self.save_model_path}")

        torch.jit.save(self.exported_model, self.save_model_path)



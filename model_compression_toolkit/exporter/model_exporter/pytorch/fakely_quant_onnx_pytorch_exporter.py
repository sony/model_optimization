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
import os
import tempfile
from typing import Callable

import torch.nn

import onnx

from model_compression_toolkit.core.common import Logger
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor
from model_compression_toolkit.exporter.model_exporter.pytorch.base_pytorch_exporter import BasePyTorchExporter


class FakelyQuantONNXPyTorchExporter(BasePyTorchExporter):
    """
    Exporter for fakely-quant PyTorch models.
    The exporter expects to receive an exportable model (where each layer's full quantization parameters
    can be retrieved), and convert it into a fakely-quant model (namely, weights that are in fake-quant
    format) and fake-quant layers for the activations.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 is_layer_exportable_fn: Callable,
                 repr_dataset: Callable):

        super().__init__(model,
                         is_layer_exportable_fn,
                         repr_dataset)

        _, self.__tmp_path_onnx = tempfile.mkstemp('.onnx')

    def export(self) -> torch.nn.Module:
        """
        Convert an exportable (fully-quantized) PyTorch model to a fakely-quant model
        (namely, weights that are in fake-quant format) and fake-quant layers for the activations.

        Returns:
            Fake-quant PyTorch model.
        """
        # assert self.is_layer_exportable_fn(layer), f'Layer {layer.name} is not exportable.'
        model_input = to_torch_tensor(next(self.repr_dataset())[0])
        torch.onnx.export(self.model,
                          model_input,
                          self.__tmp_path_onnx,
                          opset_version=13,
                          verbose=False,
                          input_names=['input'],
                          output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'},
                                        'output': {0: 'batch_size'}})
        _loaded_model = onnx.load(self.__tmp_path_onnx)
        print(f'TMP onnx path: {self.__tmp_path_onnx}')
        # os.remove(self.__tmp_path_onnx)
        # Check that the model is well formed
        onnx.checker.check_model(_loaded_model)
        self.exported_model = _loaded_model
        return self.exported_model

    def save_model(self, save_model_path: str) -> None:
        """
        Save exported model to a given path.
        Args:
            save_model_path: Path to save the model.

        Returns:
            None.
        """
        if self.exported_model is None:
            Logger.critical(f'Exporter can not save model as it is not exported')

        Logger.info(f"Exporting PyTorch fake quant onnx model: {save_model_path}")
        onnx.save_model(self.exported_model, save_model_path)


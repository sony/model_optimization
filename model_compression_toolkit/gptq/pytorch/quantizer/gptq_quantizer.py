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
import torch
import torch.nn as nn
from typing import List, Union, Dict, Any
from abc import abstractmethod
from model_compression_toolkit.core.common import Logger


class BaseWeightQuantizer(nn.Module):

    def __init__(self):
        """
        Construct a Base Pytorch model that utilizes a fake weight quantizer
        """
        super().__init__()
        self.trainable_params = dict()

    def get_trainable_params(self) -> List:
        """
        A function to get a list of trainable parameters of the quantizer for GPTQ retraining
        Returns:
            A list of trainable tensors
        """
        return [value for value in self.trainable_params.values() if value is not None]

    @abstractmethod
    def get_aux_variable(self) -> torch.Tensor:
        """
        Returns auxiliary trainable variables
        """
        raise Logger.error(f'{self.__class__.__name__} have to implement the this abstract function.')

    @abstractmethod
    def get_quantization_variable(self) -> Union[torch.Tensor, List]:
        """
        Returns quantization trainable variables
        """
        raise Logger.error(f'{self.__class__.__name__} have to implement the this abstract function.')


    @abstractmethod
    def get_weight_quantization_params(self) -> Dict[str, Any]:
        """
        Returns weight quantization dictionary params
        """
        raise Logger.error(f'{self.__class__.__name__} have to implement the this abstract function.')

    @abstractmethod
    def forward(self, w:nn.parameter, training:bool = True) -> torch.Tensor:
        """
        Forward-Pass
        Args:
            w: weights to quantize.
            training: whether in training mode or not
        Returns:
            quantized weights
        """
        raise Logger.error(f'{self.__class__.__name__} have to implement the this abstract function.')
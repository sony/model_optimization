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
from typing import Union, List

from model_compression_toolkit.logger import Logger
from model_compression_toolkit.constants import FOUND_TORCH
from model_compression_toolkit.trainable_infrastructure.common.base_trainable_quantizer import VariableGroup
from model_compression_toolkit.trainable_infrastructure.common.base_trainable_quantizer import BaseTrainableQuantizer, VAR, GROUP
from model_compression_toolkit.trainable_infrastructure import TrainableQuantizerWeightsConfig, \
    TrainableQuantizerActivationConfig


if FOUND_TORCH:

    import torch

    class BasePytorchTrainableQuantizer(BaseTrainableQuantizer):
        def __init__(self,
                     quantization_config: Union[TrainableQuantizerWeightsConfig, TrainableQuantizerActivationConfig]):
            """
            This class is a base Pytorch quantizer which validates the provided quantization config and defines an
            abstract function which any quantizer needs to implement.

            Args:
                quantization_config: quantizer config class contains all the information about the quantizer configuration.
            """
            super().__init__(quantization_config)

        def get_trainable_variables(self, group: VariableGroup) -> List[torch.Tensor]:
            """
            Get trainable parameters with specific group from quantizer

            Args:
                group: Enum of variable group

            Returns:
                List of trainable variables
            """
            quantizer_trainable = []
            for name, parameter_dict in self.quantizer_parameters.items():
                quantizer_parameter, parameter_group = parameter_dict[VAR], parameter_dict[GROUP]
                if quantizer_parameter.requires_grad and parameter_group == group:
                    quantizer_trainable.append(quantizer_parameter)
            return quantizer_trainable

else:
    class BasePytorchTrainableQuantizer(BaseTrainableQuantizer):
        def __init__(self, *args, **kwargs):
            Logger.critical("PyTorch must be installed to use 'BasePytorchTrainableQuantizer'. "
                            "The 'torch' package is missing.")  # pragma: no cover


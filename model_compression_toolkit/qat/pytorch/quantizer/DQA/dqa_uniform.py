# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
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
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from mct_quantizers.pytorch.quantizers import WeightsUniformInferableQuantizer
from torch import Tensor

from model_compression_toolkit.target_platform_capabilities.target_platform import QuantizationMethod
from mct_quantizers import QuantizationTarget, PytorchQuantizationWrapper, mark_quantizer
from model_compression_toolkit.constants import RANGE_MAX, RANGE_MIN
from model_compression_toolkit.trainable_infrastructure import TrainingMethod
from model_compression_toolkit.qat.pytorch.quantizer.base_pytorch_qat_weight_quantizer import BasePytorchQATWeightTrainableQuantizer
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor
from model_compression_toolkit.trainable_infrastructure.pytorch.quantizer_utils import uniform_quantizer
from model_compression_toolkit.trainable_infrastructure import TrainableQuantizerWeightsConfig
from model_compression_toolkit.trainable_infrastructure.common.constants import FQ_MIN, FQ_MAX


@mark_quantizer(quantization_target=QuantizationTarget.Weights,
                quantization_method=[QuantizationMethod.UNIFORM],
                identifier=TrainingMethod.DQA)
class DQAUniformWeightQuantizer(BasePytorchQATWeightTrainableQuantizer):
    """
    Trainable constrained quantizer to quantize a layer inputs.
    """

    def __init__(self, quantization_config: TrainableQuantizerWeightsConfig):
        """
        Initialize a TrainableWeightQuantizer object with parameters to use
        for the quantization.

        Args:
            quantization_config: trainable quantizer config class
        """
        super().__init__(quantization_config)
        self.num_bits = self.quantization_config.weights_n_bits
        self.min_int = 0
        self.max_int = 2 ** self.num_bits - 1
        self.max_values = quantization_config.weights_quantization_params[RANGE_MAX]
        self.min_values = quantization_config.weights_quantization_params[RANGE_MIN]
        self.min_max_shape = np.asarray(self.max_values).shape
        self.max = np.reshape(self.max_values,
                              [-1]) if self.quantization_config.weights_per_channel_threshold else float(
            self.max_values)
        self.min = np.reshape(self.min_values,
                              [-1]) if self.quantization_config.weights_per_channel_threshold else float(
            self.min_values)
        self.bitwidth_option=[self.num_bits, 2*self.num_bits,4*self.num_bits]

        self.quantizer_parameters = {}
        self.Attention = "attention"
        self.Temperature = "temperature"

    def initialize_quantization(self,
                                tensor_shape: torch.Size,
                                name: str,
                                layer: PytorchQuantizationWrapper) -> Dict[str, nn.Parameter]:
        """
        Add min and max variables to layer.
        Args:
            tensor_shape: Tensor shape the quantizer quantize.
            name: Prefix of variables names.
            layer: Layer to add the variables to. The variables are saved
            in the layer's scope.

        Returns:
            Dictionary of new variables.
        """
        # define temperature and attention values
        layer.register_parameter(name+"_"+self.Temperature, torch.zero(1))
        layer.register_parameter(name+"_"+self.Attention, nn.Parameter(torch.FloatTensor([4/7,2/7,1/7])))
        #layer.register_parameter(name+"_"+'Attention_factor', nn.Parameter(torch.FloatTensor([1,4,8]))) # should be done with the loss function for training.
        

        # Add min and max variables to layer.
        layer.register_parameter(name+"_"+FQ_MIN, nn.Parameter(to_torch_tensor(self.min_values), requires_grad=False))
        layer.register_parameter(name+"_"+FQ_MAX, nn.Parameter(to_torch_tensor(self.max_values), requires_grad=False))

        # Save the quantizer parameters for later calculations
        self.quantizer_parameters = {FQ_MIN: layer.get_parameter(name+"_"+FQ_MIN), FQ_MAX: layer.get_parameter(name+"_"+FQ_MAX), 
        self.Attention:layer.get_parameter(name+"_"+self.Attention),self.Temperature:layer.get_parameter(name+"_"+self.Temperature)}

        return self.quantizer_parameters

    def __call__(self,
                 inputs: nn.Parameter,
                 training: bool) -> Tensor:
        """
        Quantize a tensor
        Args:
            inputs: Input tensor to quantize.
            training: whether in training mode or not
        Returns:
            quantized tensor
        """
        attention = self.quantizer_parameters[self.Attention] / self.quantizer_parameters[self.Attention].std() 
        
        attention = torch.nn.functional.softmax(self.quantizer_parameters[self.Temperature] * attention)
        q=0
        for i in range(len(self.bitwidth_option)):
            q+= attention[i]*uniform_quantizer(inputs, self.quantizer_parameters[FQ_MIN], self.quantizer_parameters[FQ_MAX], self.bitwidth_option[i])

        return q

    def convert2inferable(self) -> WeightsUniformInferableQuantizer:
        """
        Convert quantizer to inferable quantizer.

        Returns:
            A pytorch inferable quanizer object.
        """
        _min = self.quantizer_parameters[FQ_MIN].cpu().detach().numpy()
        _max = self.quantizer_parameters[FQ_MAX].cpu().detach().numpy()

        return WeightsUniformInferableQuantizer(num_bits=self.num_bits,
                                                   min_range=_min, max_range=_max,
                                                   per_channel=self.quantization_config.weights_per_channel_threshold,
                                                   channel_axis=self.quantization_config.weights_channels_axis)



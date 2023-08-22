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
from typing import Type, Dict, Tuple, List
import torch
from torch import Tensor
from torch.nn import Module

from model_compression_toolkit.core.pytorch.utils import get_working_device
from model_compression_toolkit.data_generation.common.model_info_exctractors import OriginalBNStatsHolder, \
    ActivationExtractor
from model_compression_toolkit.data_generation.pytorch.constants import IMAGE_INPUT, NUM_INPUT_CHANNELS


class PytorchOriginalBNStatsHolder(OriginalBNStatsHolder):
    """
    Holds the original batch normalization (BN) statistics for a model.
    """
    def __init__(self,
                 model: Module,
                 bn_layer_types: List):
        """
        Initializes the PytorchOriginalBNStatsHolder.

        Args:
            model (Module): The PyTorch model.
            bn_layer_types (List): List of batch normalization layer types.
        """
        self.device = get_working_device()
        super(PytorchOriginalBNStatsHolder, self).__init__(model, bn_layer_types)

    def get_bn_params(self,
                      model: Module,
                      bn_layer_types: List) -> Dict[str, Tuple[Tensor, Tensor, Tensor]]:
        """
        Get the batch normalization parameters (mean and variance) for each batch normalization layer in the model.

        Args:
            model (Module): The PyTorch model.
            bn_layer_types (List): List of batch normalization layer types.

        Returns:
            dict: Dictionary mapping batch normalization layer names to their parameters.
        """
        bn_params = {}
        # Assume the images in the dataset are normalized to be 0-mean, 1-variance
        imgs_mean = torch.zeros(1, NUM_INPUT_CHANNELS).to(self.device)
        imgs_var = torch.ones(1, NUM_INPUT_CHANNELS).to(self.device)
        bn_params.update({IMAGE_INPUT: (imgs_mean, imgs_var, imgs_var)})
        for name, module in model.named_modules():
            if isinstance(module, tuple(bn_layer_types)):
                mean = module.running_mean.detach().clone().flatten().to(self.device)
                var = module.running_var.detach().clone().flatten().to(self.device)
                std = torch.sqrt(var)
                bn_params.update({name: (mean, var, std)})
        return bn_params


class InputHook:
    """
    Forward_hook used to extract the input of an intermediate layer.
    """
    def __init__(self):
        """
        Initialize the InputHook.

        """
        super(InputHook, self).__init__()
        self.input = None

    def hook(self,
             module: Module,
             input: Tensor,
             output: Tensor):
        """
        Hook function to extract the input of the layer.

        Args:
            module (Module): Pytorch module.
            input (Tensor): Input tensor.
            output (Tensor): Output tensor.
        """
        self.input = input[0]

    def clear(self):
        """
        Clear the stored input tensor.
        """
        self.input = None


class PytorchActivationExtractor(ActivationExtractor):
    """
    Extracts activations of inputs to layers in a model using PyTorch hooks.
    """
    def __init__(self,
                 model: Module,
                 layer_types_to_extract_inputs: List):
        """
        Initializes the PytorchActivationExtractor.

        Args:
            model (Module): The PyTorch model.
            layer_types_to_extract_inputs (List): Tuple or list of layer types.
        """
        self.model = model
        self.num_layers = sum([1 if isinstance(layer, tuple(layer_types_to_extract_inputs)) else 0 for layer in model.modules()])
        print(f'Number of layers = {self.num_layers}')
        self.hooks = {}  # Dictionary to store InputHook instances by layer name
        self.hook_handles = []  # List to store hook handles
        for name, module in model.named_modules():
            if isinstance(module, tuple(layer_types_to_extract_inputs)):
                hook = InputHook()# Create an InputHook instance
                self.hooks.update({name: hook})
                hook_handle = module.register_forward_hook(hook.hook)# Register the hook on the module
                self.hook_handles.append(hook_handle)# Store the hook handle in the hook_handles list

    def get_layer_input_activation(self, layer_name: str) -> Tensor:
        """
        Get the input activation tensor of a layer.

        Args:
            layer_name (str): Name of the layer.

        Returns:
            Tensor: Input activation tensor of the layer.
        """
        return self.hooks.get(layer_name).input

    def get_num_extractor_layers(self) -> int:
        """
        Get the number of hooked layers in the model.

        Returns:
            int: Number of hooked layers in the model.
        """
        return self.num_layers

    def get_extractor_layer_names(self) -> list:
        """
        Get a list of the hooked layer names.

        Returns:
            list: A list of the hooked layer names.
        """
        return list(self.hooks.keys())

    def clear(self):
        """
        Clear the stored activation tensors.
        """
        for hook in self.hooks:
            hook.clear()

    def remove(self):
        """
        Remove the hooks from the model.
        """
        self.clear()
        for handle in self.hook_handles:
            handle.remove()

    def run_model(self, inputs: Tensor) -> Tensor:
        """
        Run the model on the given inputs and return the output.

        Args:
            inputs (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        return self.model(inputs)



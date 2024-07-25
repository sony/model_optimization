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
from typing import Dict, Tuple, List
import torch
from torch import Tensor
from torch.fx import GraphModule
from torch.nn import Module, Conv2d, Linear

from model_compression_toolkit.core.pytorch.pytorch_device_config import get_working_device
from model_compression_toolkit.data_generation.common.model_info_exctractors import OriginalBNStatsHolder, \
    ActivationExtractor
from model_compression_toolkit.data_generation.pytorch.constants import OUTPUT
from model_compression_toolkit.data_generation.common.constants import IMAGE_INPUT, NUM_INPUT_CHANNELS
from model_compression_toolkit.logger import Logger


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
                 fx_model: GraphModule,
                 layer_types_to_extract_inputs: List,
                 last_layer_types_to_extract_inputs: List):
        """
        Initializes the PytorchActivationExtractor.

        Args:
            model (Module): The PyTorch model.
            fx_model (GraphModule): A static graph representation of the PyTorch model.
            layer_types_to_extract_inputs (List): Tuple or list of layer types.
            last_layer_types_to_extract_inputs (List): Tuple or list of layer types.
        """
        self.model = model
        self.fx_model = fx_model
        self.layer_types_to_extract_inputs = tuple(layer_types_to_extract_inputs)
        self.last_layer_types_to_extract_inputs = tuple(last_layer_types_to_extract_inputs)
        self.num_layers = sum([1 if isinstance(layer, tuple(layer_types_to_extract_inputs)) else 0 for layer in model.modules()])
        self.hooks = {}  # Dictionary to store InputHook instances by layer name
        self.last_linear_layers_hooks = {}  # Dictionary to store InputHook instances by layer name
        self.hook_handles = []  # List to store hook handles
        self.last_linear_layer_weights = [] # list of the last linear layers' weights

        # set hooks for batch norm layers
        self._set_hooks_for_layers()

        # set hooks for last output layers
        self._set_hooks_for_last_layers()

    def _set_hooks_for_layers(self):
        """
        This function sets hooks for the inputs of layers of type "self.layer_types_to_extract_inputs"
        """
        for name, module in self.model.named_modules():
            if isinstance(module, self.layer_types_to_extract_inputs):
                hook = InputHook()# Create an InputHook instance
                self.hooks.update({name: hook})
                hook_handle = module.register_forward_hook(hook.hook)# Register the hook on the module
                self.hook_handles.append(hook_handle)# Store the hook handle in the hook_handles list


    def _set_hooks_for_last_layers(self):
        """
        This function finds the output layers of the model and adds hooks to the input activation of
        those layers.
        """
        for node in self.fx_model.graph.nodes:

            # Find the output nodes in the graph
            if node.op == 'output':
                found_linear_node = False
                nodes_to_search = node.all_input_nodes

                # Search graph from the output node and back until we find a linear node
                for node_to_search in nodes_to_search:
                    for name, module in self.model.named_modules():
                        if name == node_to_search.target:
                            if isinstance(module, Linear) or isinstance(module, Conv2d):
                                self.last_linear_layer_weights.append(module.weight.data.clone())
                                hook = InputHook()  # Create an InputHook instance
                                self.last_linear_layers_hooks.update({name: hook})
                                hook_handle = module.register_forward_hook(hook.hook)  # Register the hook on the module
                                self.hook_handles.append(hook_handle)
                                found_linear_node = True
                    if not found_linear_node:
                        nodes_to_search += node_to_search.all_input_nodes

    def get_layer_input_activation(self, layer_name: str) -> Tensor:
        """
        Get the input activation tensor of a layer.

        Args:
            layer_name (str): Name of the layer.

        Returns:
            Tensor: Input activation tensor of the layer.
        """
        return self.hooks.get(layer_name).input

    def get_output_layer_input_activation(self) -> List:
        """
        Get the input activation tensors of all the output layers that are Linear or Conv2d.

        Returns:
            List: Input activation tensors of all the output layers that are Linear or Conv2d.
        """
        return [v.input for v in self.last_linear_layers_hooks.values()]

    def get_last_linear_layers_weights(self) -> List:
        """
        Get the weight tensors of all the last linear layers.

        Returns:
            List: Weight tensors of all the last linear layers.
        """
        return self.last_linear_layer_weights

    def get_extractor_layer_names(self) -> List:
        """
        Get a list of the hooked layer names.

        Returns:
            List: A list of the hooked layer names.
        """
        return list(self.hooks.keys())

    def clear(self):
        """
        Clear the stored activation tensors.
        """
        for hook in self.hooks:
            hook.clear()
        for hook in self.last_linear_layers_hooks:
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



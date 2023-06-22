from typing import Type
import torch
from torch import Tensor
from torch.nn import Module
from model_compression_toolkit.data_generation.pytorch.constants import DEVICE, IMAGE_INPUT


class OrigBNStatsHolder(object):
    """
    Holds the original batch normalization (BN) statistics for a model.
    """
    def __init__(self,
                 model: Module,
                 bn_layer_types: Type[list],
                 eps=1e-6):
        """
        Initializes the OrigBNStatsHolder.

        Args:
            model (Module): The PyTorch model.
            bn_layer_types (Type[list]): List of batch normalization layer types.
            eps (float): Epsilon value for numerical stability.
        """
        self.bn_params = self.get_bn_params(model, bn_layer_types)
        self.eps = eps

    def get_bn_layer_names(self):
        """
        Get the names of all batch normalization layers.

        Returns:
            list: List of batch normalization layer names.
        """
        return list(self.bn_params.keys())

    def get_mean(self, bn_layer_name: str):
        """
        Get the mean of a batch normalization layer.

        Args:
            bn_layer_name (str): Name of the batch normalization layer.

        Returns:
            Tensor: Mean of the batch normalization layer.
        """
        return self.bn_params[bn_layer_name][0]

    def get_var(self, bn_layer_name: str):
        """
        Get the variance of a batch normalization layer.

        Args:
            bn_layer_name (str): Name of the batch normalization layer.

        Returns:
            Tensor: Variance of the batch normalization layer.
        """
        return self.bn_params[bn_layer_name][1]

    def get_std(self, bn_layer_name: str):
        """
        Get the standard deviation of a batch normalization layer.

        Args:
            bn_layer_name (str): Name of the batch normalization layer.

        Returns:
            Tensor: Standard deviation of the batch normalization layer.
        """
        var = self.bn_params[bn_layer_name][1]
        std = torch.sqrt(var + self.eps)
        return std

    def get_num_bn_layers(self):
        """
        Get the number of batch normalization layers.

        Returns:
            int: Number of batch normalization layers.
        """
        return len(self.bn_params)

    @staticmethod
    def get_bn_params(model: Module,
                      bn_layer_types: Type[list]):
        """
        Get the batch normalization parameters (mean and variance) for each batch normalization layer in the model.

        Args:
            model (Module): The PyTorch model.
            bn_layer_types (Type[list]): List of batch normalization layer types.

        Returns:
            dict: Dictionary mapping batch normalization layer names to their parameters.
        """
        bn_params = {}
        # Assume the images in the dataset are normalized to be 0-mean, 1-variance
        imgs_mean = torch.zeros(1, 3).to(DEVICE)
        imgs_var = torch.ones(1, 3).to(DEVICE)
        bn_params.update({IMAGE_INPUT: (imgs_mean, imgs_var)})
        for name, module in model.named_modules():
            if isinstance(module, tuple(bn_layer_types)):
                mean = module.running_mean.detach().clone().flatten().to(DEVICE)
                var = module.running_var.detach().clone().flatten().to(DEVICE)
                bn_params.update({name: (mean, var)})
        return bn_params


class BatchNormInputHook(object):
    """
    Forward_hook used to extract the input of an intermediate batch norm layer.
    """
    def __init__(self):
        """
        Initialize the BatchNormInputHook.

        """
        super(BatchNormInputHook, self).__init__()
        self.input = None

    def hook(self,
             module: Module,
             input: Tensor,
             output: Tensor):
        """
        Hook function to extract the input of the batch normalization layer.

        Args:
            module (Module): Batch normalization module.
            input (Tensor): Input tensor.
            output (Tensor): Output tensor.
        """
        self.input = input[0]

    def clear(self):
        """
        Clear the stored input tensor.
        """
        self.input = None


class ActivationExtractor(object):
    """
    Extracts activations of inputs to batch normalization layers in a model.
    """
    def __init__(self,
                 model: Module,
                 bn_layer_types: Type[list]):
        """
        Initializes the ActivationExtractor.

        Args:
            model (Module): The PyTorch model.
            bn_layer_types (Type[list]): Tuple or list of batch normalization layer types.
        """
        self.model = model
        self.num_bn_layers = sum([1 if isinstance(layer, tuple(bn_layer_types)) else 0 for layer in model.modules()])
        print(f'Number of BatchNorm layers = {self.num_bn_layers}')
        self.hooks = {}  # Dictionary to store BatchNormInputHook instances by layer name
        self.hook_handles = []  # List to store hook handles
        for name, module in model.named_modules():
            if isinstance(module, tuple(bn_layer_types)):
                bn_hook = BatchNormInputHook()# Create a BatchNormInputHook instance
                self.hooks.update({name: bn_hook})
                hook_handle = module.register_forward_hook(bn_hook.hook)# Register the hook on the module
                self.hook_handles.append(hook_handle)# Store the hook handle in the hook_handles list

    def get_activation(self, layer_name: str) -> Tensor:
        """
        Get the activation (input) tensor of a batch normalization layer.

        Args:
            layer_name (str): Name of the batch normalization layer.

        Returns:
            Tensor: Activation tensor of the batch normalization layer.
        """
        return self.hooks.get(layer_name).input

    def get_num_bn_layers(self) -> int:
        """
        Get the number of batch normalization layers.

        Returns:
            int: Number of batch normalization layers.
        """
        return self.num_bn_layers

    def get_bn_layer_names(self) -> list:
        """
        Get a list of batch normalization layer names.

        Returns:
            list: A list of batch normalization layer names.
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

    def run_on_inputs(self, inputs: Tensor) -> Tensor:
        """
        Run the model on the given inputs and return the output.

        Args:
            inputs (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        return self.model(inputs)



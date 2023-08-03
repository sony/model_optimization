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
from typing import Type, Any, List


class OrigBNStatsHolder(object):
    """
    Holds the original batch normalization (BN) statistics for a model.
    """
    def __init__(self,
                 model: Any,
                 bn_layer_types: Type[list],
                 eps=1e-6):
        """
        Initializes the OrigBNStatsHolder.

        Args:
            model (Any): The framework model.
            bn_layer_types (Type[list]): List of batch normalization layer types.
            eps (float): Epsilon value for numerical stability.
        """
        self.eps = eps
        self.bn_params = self.get_bn_params(model, bn_layer_types)

    def get_bn_layer_names(self) -> List[str]:
        """
        Get the names of all batch normalization layers.

        Returns:
            List[str]: List of batch normalization layer names.
        """
        return list(self.bn_params.keys())

    def get_mean(self, bn_layer_name: str) -> Any:
        """
        Get the mean of a batch normalization layer.

        Args:
            bn_layer_name (str): Name of the batch normalization layer.

        Returns:
            Any: Mean of the batch normalization layer.
        """
        return self.bn_params[bn_layer_name][0]

    def get_var(self, bn_layer_name: str) -> Any:
        """
        Get the variance of a batch normalization layer.

        Args:
            bn_layer_name (str): Name of the batch normalization layer.

        Returns:
            Any: Variance of the batch normalization layer.
        """
        return self.bn_params[bn_layer_name][1]

    def get_std(self, bn_layer_name: str) -> Any:
        """
        Get the standard deviation of a batch normalization layer.

        Args:
            bn_layer_name (str): Name of the batch normalization layer.

        Returns:
            Any: Standard deviation of the batch normalization layer.
        """
        return self.bn_params[bn_layer_name][2]

    def get_num_bn_layers(self) -> int:
        """
        Get the number of batch normalization layers.

        Returns:
            int: Number of batch normalization layers.
        """
        return len(self.bn_params)

    def get_bn_params(self,
                      model: Any,
                      bn_layer_types: Type[list]) -> dict:
        """
        Get the batch normalization parameters (mean and variance) for each batch normalization layer in the model.

        Args:
            model (Any): The model.
            bn_layer_types (Type[list]): List of batch normalization layer types.

        Returns:
            dict: Dictionary mapping batch normalization layer names to their parameters.
        """
        raise NotImplemented



class ActivationExtractor(object):
    """
    Extracts activations of inputs to layers in a model.
    """
    def __init__(self,
                 model: Any,
                 layer_types_to_extract_inputs: Type[list]):
        """
        Initializes the ActivationExtractor.

        Args:
            model (Any): The model.
            layer_types_to_extract_inputs (Type[list]): Tuple or list of layer types.
        """
        raise NotImplemented

    def get_activation(self, layer_name: str) -> Any:
        """
        Get the activation (input) tensor of a layer.

        Args:
            layer_name (str): Name of the layer.

        Returns:
            Any: Activation tensor of the layer.
        """
        raise NotImplemented

    def get_num_extractor_layers(self) -> int:
        """
        Get the number of layers for which to extract input activations.

        Returns:
            int: Number of layers for which to extract input activations.
        """
        raise NotImplemented

    def get_extractor_layer_names(self) -> list:
        """
        Get a list of the layer names for which to extract input activations.

        Returns:
            list: A list of layer names for which to extract input activations.
        """
        raise NotImplemented

    def clear(self):
        """
        Clear the stored activation tensors.
        """
        raise NotImplemented

    def remove(self):
        """
        Remove the hooks from the model.
        """
        raise NotImplemented

    def run_on_inputs(self, inputs: Any) -> Any:
        """
        Run the model on the given inputs and return the output.

        Args:
            inputs (Any): Input tensor.

        Returns:
            Any: Output tensor.
        """
        raise NotImplemented



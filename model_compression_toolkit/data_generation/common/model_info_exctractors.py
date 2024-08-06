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
from abc import abstractmethod
from typing import Any, List, Dict, Tuple

from model_compression_toolkit.logger import Logger


class OriginalBNStatsHolder:
    """
    Holds the original batch normalization (BN) statistics for a model.
    """

    def __init__(self,
                 model: Any,
                 bn_layer_types: List):
        """
        Initializes the OriginalBNStatsHolder.

        Args:
            model (Any): The framework model.
            bn_layer_types (List): List of batch normalization layer types.
        """
        self.bn_params = self.get_bn_params(model, bn_layer_types)
        if self.get_num_bn_layers() == 0:
            Logger.critical(f'Data generation requires a model with at least one BatchNorm layer.')

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
                      bn_layer_types: List) -> Dict[str, Tuple]:
        """
        Get the batch normalization parameters (mean and variance) for each batch normalization layer in the model.

        Args:
            model (Any): The model.
            bn_layer_types (List): List of batch normalization layer types.

        Returns:
            dict: Dictionary mapping batch normalization layer names to their parameters.
        """
        raise NotImplemented # pragma: no cover


class ActivationExtractor:
    """
    Extracts activations of input tensors to layers in a model.
    """

    def __init__(self,
                 model: Any,
                 layer_types_to_extract_inputs: List):
        """
        Initializes the ActivationExtractor.

        Args:
            model (Any): The model.
            layer_types_to_extract_inputs (List): Tuple or list of layer types.
        """
        raise NotImplemented # pragma: no cover

    @abstractmethod
    def get_layer_input_activation(self, layer_name: str) -> Any:
        """
        Get the input activation tensor of a layer.

        Args:
            layer_name (str): Name of the layer.

        Returns:
            Any: Input activation tensor of the layer.
        """
        raise NotImplemented # pragma: no cover

    @abstractmethod
    def get_output_layer_input_activation(self) -> List:
        """
        Get the input activation tensors of all the output layers that are Linear or Conv2d.

        Returns:
            Any: Input activation tensors of all the output layers that are Linear or Conv2d.
        """
        raise NotImplemented # pragma: no cover

    @abstractmethod
    def get_last_linear_layers_weights(self) -> List:
        """
        Get the weight tensors of all the last linear layers.

        Returns:
            List: Weight tensors of all the last linear layers.
        """
        raise NotImplemented # pragma: no cover


    @abstractmethod
    def get_extractor_layer_names(self) -> List:
        """
        Get a list of the layer names for which to extract input activations.

        Returns:
            List: A list of layer names for which to extract input activations.
        """
        raise NotImplemented # pragma: no cover

    @abstractmethod
    def clear(self):
        """
        Clear the stored activation tensors.
        """
        raise NotImplemented # pragma: no cover

    @abstractmethod
    def remove(self):
        """
        Remove the hooks from the model.
        """
        raise NotImplemented # pragma: no cover

    @abstractmethod
    def run_model(self, inputs: Any) -> Any:
        """
        Run the model on the given inputs and return the output.

        Args:
            inputs (Any): Input tensor.

        Returns:
            Any: Output tensor.
        """
        raise NotImplemented # pragma: no cover

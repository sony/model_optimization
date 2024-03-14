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
from typing import List, Dict, Tuple, Callable

import tensorflow as tf
from keras.layers import Dense, Conv2D
from tensorflow.keras.layers import BatchNormalization

from model_compression_toolkit.data_generation.common.enums import ImageGranularity
from model_compression_toolkit.data_generation.common.model_info_exctractors import OriginalBNStatsHolder, \
    ActivationExtractor
from model_compression_toolkit.logger import Logger


class KerasOriginalBNStatsHolder(OriginalBNStatsHolder):
    """
    Holds the original batch normalization (BN) statistics for a model.
    """

    def __init__(self,
                 model: tf.keras.Model,
                 bn_layer_types: List = [BatchNormalization]):
        """
        Initializes the KerasOriginalBNStatsHolder.

        Args:
            model (Model): Keras model to generate data for.
            bn_layer_types (List): List of batch normalization layer types.
        """
        super(KerasOriginalBNStatsHolder, self).__init__(model, bn_layer_types)

    def get_bn_params(self,
                      model: tf.keras.Model,
                      bn_layer_types: List) -> Dict[str, Tuple[tf.Tensor, tf.Tensor]]:
        """
        Get the batch normalization parameters (mean and variance) for each batch normalization layer in the model.

        Args:
            model (Model): Keras model to generate data for.
            bn_layer_types (List): List of batch normalization layer types.

        Returns:
            dict: Dictionary mapping batch normalization layer names to their parameters.
        """
        bn_params = {}
        for layer in model.layers:
            if isinstance(layer, tuple(bn_layer_types)):
                layer_name = layer.name
                mean = layer.moving_mean.numpy().flatten()
                variance = layer.moving_variance.numpy().flatten()
                bn_params[layer_name] = (mean, variance)
        return bn_params


class KerasActivationExtractor(ActivationExtractor):
    """
    Extracts activations of inputs to layers in a model using PyTorch hooks.
    """

    def __init__(self,
                 model: tf.keras.Model,
                 layer_types_to_extract_inputs: List,
                 image_granularity: ImageGranularity,
                 image_input_manipulation: Callable,
                 linear_layers: Tuple = (Dense, Conv2D)):
        """
        Initializes the KerasActivationExtractor.

        Args:
            model (Model): Keras model to generate data for.
            layer_types_to_extract_inputs (List): Tuple or list of layer types.
            image_granularity (ImageGranularity): The granularity of the images for optimization.
            image_input_manipulation (Callable): Function for image input manipulation.
            linear_layers (Tuple): Tuple of linear layers types to retrieve the output of the last linear layer

        """
        self.model = model
        self.image_input_manipulation = image_input_manipulation
        self.image_granularity = image_granularity
        self.layer_types_to_extract_inputs = tuple(layer_types_to_extract_inputs)
        self.linear_layers = linear_layers

        # Create a list of BatchNormalization layer names from the model.
        self.bn_layer_names = [layer.name for layer in model.layers if isinstance(layer,
                                                                                  self.layer_types_to_extract_inputs)]
        self.num_layers = len(self.bn_layer_names)
        Logger.info(f'Number of layers = {self.num_layers}')

        # Initialize stats containers
        self.activations = {}

        # Initialize the last linear layer output variable as None In case the last layer is a linear layer (conv,
        # dense) last_linear_layer_output will assign with the last linear layer output value, if not the value will
        # stay None
        self.last_linear_layer_output = None

        # Get the last layer, if the last layer is linear (conv, dense)
        self.last_linear_layers = self.get_model_last_layer()

        # Create list for the layers we use for optimization
        self.layer_list = [layer for layer in self.model.layers if
                           isinstance(layer, self.layer_types_to_extract_inputs)]

        # Create list of outputs to the intermediate model
        # We want the input of each BN layer
        self.outputs_list = [layer.input for layer in self.layer_list]

        # If the last layer is linear add the output of the layer to the layers to optimize
        if self.last_linear_layers is not None:
            self.layer_list.append(self.last_linear_layers)
            if self.last_linear_layers is not self.model.layers[-1]:
                self.outputs_list.append(self.last_linear_layers.output)
        self.outputs_list.append(self.model.output)

        # Create an intermediate model with the outputs defined before.
        self.intermediate_model = tf.keras.Model(inputs=self.model.input,
                                                 outputs=self.outputs_list)

    def get_layer_input_activation(self,
                                   layer_name: str) -> Dict:
        """
        Get the activation data for a specific layer.

        Args:
            layer_name: Name of the layer to retrieve activation data for.

        Returns:
            Layer and input data for the layer
        """
        return self.activations.get(layer_name)

    def get_extractor_layer_names(self) -> List:
        """
        Get a list of the bn layer names for which to extract input activations.

        Returns:
            List: A list of bn layer names for which to extract input activations.
        """
        return self.bn_layer_names

    @tf.function
    def run_on_inputs(self,
                      inputs: tf.Tensor) -> List[tf.Tensor]:
        """
        Run the model on the given inputs and return the intermediate outputs,
        wrapped by a tf.function for acceleration.

        Args:
            inputs (Tensor): Input tensor.

        Returns:
            List[Tensor]: Intermediate output tensors.
        """
        return self.intermediate_model(inputs=inputs)

    def run_model(self,
                  inputs: tf.Tensor) -> tf.Tensor:
        """
        Run the model on the given inputs and return the output.

        Args:
            inputs (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """

        # Run the model on inputs
        intermediate_outputs = self.run_on_inputs(inputs=inputs)

        # Iterate over layers to extract stats
        for i, layer in enumerate(self.layer_list):
            if isinstance(layer, self.layer_types_to_extract_inputs):
                input_data = intermediate_outputs[i]
                # Save the layer and its input data in the dictionary
                self.activations[layer.name] = {'layer': layer, 'input_data': input_data}
            elif layer == self.last_linear_layers:
                self.last_linear_layer_output = intermediate_outputs[i]
        # Last intermediate output is the output of the model
        return intermediate_outputs[-1]

    def get_model_last_layer(self):
        """
        Get the last layer in the model that is not one of the specified layer types.

        Returns:
            The last layer in the model that meets the criteria, or None if not found.
        """
        last_layer = None
        for layer in reversed(self.model.layers):
            if isinstance(layer, self.linear_layers):
                if not any(isinstance(node.layer, self.layer_types_to_extract_inputs)
                           for node in layer._outbound_nodes):
                    last_layer = layer
                    break
        return last_layer

    def remove(self):
        """
        Remove the stats containers.
        """
        self.activations = {}

from typing import List, Type, Dict, Tuple, Callable

import tensorflow as tf
from keras.layers import Dense, Conv2D
from tensorflow.keras.layers import BatchNormalization

from model_compression_toolkit.data_generation.common.enums import ImageGranularity
from model_compression_toolkit.data_generation.common.model_info_exctractors import OriginalBNStatsHolder, \
    ActivationExtractor
from model_compression_toolkit.data_generation.keras.constants import H_AXIS, W_AXIS, BATCH_AXIS


class KerasOriginalBNStatsHolder(OriginalBNStatsHolder):
    """
    Holds the original batch normalization (BN) statistics for a model.
    """

    def __init__(self,
                 model: tf.keras.Model,
                 bn_layer_types: Type[List] = [BatchNormalization]):
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
        self.layer_types_to_extract_inputs = layer_types_to_extract_inputs
        self.linear_layers = linear_layers

        # Create a list of BatchNormalization layer names from the model.
        self.bn_layer_names = [layer.name for layer in model.layers if isinstance(layer,
                                                                                  tuple(layer_types_to_extract_inputs))]
        self.num_bn_layers = len(self.bn_layer_names)
        print("Num. of BatchNorm layers =", self.num_bn_layers)

        # Initialize stats containers
        self.activations = {}
        self.bn_mean = []
        self.bn_var = []

        # Initialize the last linear layer output variable as None In case the last layer is a linear layer (conv,
        # dense) last_linear_layer_output will assign with the last linear layer output value, if not the value will
        # stay None
        self.last_linear_layer_output = None

        # Set the mean axis based on the image granularity
        if self.image_granularity == ImageGranularity.ImageWise:
            self.mean_axis = [H_AXIS, W_AXIS]
        else:
            self.mean_axis = [BATCH_AXIS, H_AXIS, W_AXIS]

        # Get the last layer, if the last layer is linear (conv, dense)
        self.last_linear_layer = self.get_model_last_layer()

        # Create list for the layers we use for optimization
        self.layer_list = [layer for layer in self.model.layers if
                           isinstance(layer, tuple(self.layer_types_to_extract_inputs))]

        # Create list of outputs to the intermediate model
        # We want the input of each BN layer
        self.outputs_list = [layer.input for layer in self.layer_list]

        # If the last layer is linear add the output of the layer to the layers to optimize
        if self.last_linear_layer is not None:
            self.layer_list.append(self.last_linear_layer)
            self.outputs_list.append(self.last_linear_layer.output)

        # Create an intermediate model with the outputs defined before.
        self.intermediate_model = tf.keras.Model(inputs=self.model.input,
                                                 outputs=self.outputs_list)

    def get_bn_layer_names(self):
        """
        Get a list of the bn layer names for which to extract input activations.

        Returns:
            list: A list of bn layer names for which to extract input activations.
        """
        return self.bn_layer_names

    def run_model(self,
                  inputs: tf.Tensor) -> tf.Tensor:
        """
        Run the model on the given inputs and return the output.

        Args:
            inputs (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        # Initialize stats containers
        self.bn_mean = []
        self.bn_var = []

        # Run the model on inputs
        intermediate_outputs = self.intermediate_model(inputs=inputs)

        # Iterate over layers to extract stats
        for i, layer in enumerate(self.layer_list):
            if isinstance(layer, tuple(self.layer_types_to_extract_inputs)):
                input_data = intermediate_outputs[i]
                # Save the layer and its input data in the dictionary
                self.activations[layer.name] = {'layer': layer, 'input_data': input_data}

                mean, var = tf.nn.moments(x=input_data, axes=self.mean_axis, keepdims=False)
                self.bn_mean.append(mean)
                self.bn_var.append(var)
            else:
                self.last_linear_layer_output = intermediate_outputs[i]

        # Run the model to get the output of the model
        output = self.model(inputs)
        return output

    def get_activation(self,
                       layer_name: str) -> Dict:
        """
        Get the activation data for a specific layer.

        Args:
            layer_name: Name of the layer to retrieve activation data for.

        Returns:
            Layer and input data for the layer
        """
        return self.activations.get(layer_name)

    def get_model_last_layer(self):
        """
        Get the last layer in the model that is not one of the specified layer types.

        Returns:
            The last layer in the model that meets the criteria, or None if not found.
        """
        last_layer = None
        for layer in reversed(self.model.layers):
            if isinstance(layer, self.linear_layers):
                if not any(isinstance(node.layer, tuple(self.layer_types_to_extract_inputs))
                           for node in layer._outbound_nodes):
                    last_layer = layer
                    break
        return last_layer

    def remove(self):
        """
        Remove the stats containers.
        """
        self.activations = {}
        self.bn_mean = []
        self.bn_var = []

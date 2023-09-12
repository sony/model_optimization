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
from typing import Dict, Callable
import tensorflow as tf

from model_compression_toolkit.logger import Logger
from model_compression_toolkit.data_generation.common.enums import OutputLossType
from model_compression_toolkit.data_generation.keras.model_info_exctractors import KerasActivationExtractor


# Function to calculate the regularized min-max difference loss
def regularized_min_max_diff(
        output_imgs: tf.Tensor,
        activation_extractor: KerasActivationExtractor,
        tape: tf.GradientTape,
        eps: float = 1e-6,
        **kwargs) -> tf.Tensor:
    """
    Calculate the regularized min-max difference loss.

    This function calculates the regularized min-max difference loss based on the provided inputs.

    Args:
        output_imgs (tf.Tensor): Output images or tensors.
        activation_extractor (KerasActivationExtractor): Activation extractor object.
        tape (tf.GradientTape): TensorFlow tape for recording operations.
        eps (float, optional): Small constant to prevent division by zero.
        **kwargs: Additional keyword arguments.

    Returns:
        tf.Tensor: The calculated loss.
    """
    if activation_extractor.last_linear_layers is None:
        Logger.exception(
            f'Cannot compute regularized min max output loss for the input model. The regularized min max output loss '
            f'requires linear layer without a following BatchNormalization layer. Please choose one of '
            f'{OutputLossType.get_values()}.')

    with tape.stop_recording():
        weights_last_layer = activation_extractor.last_linear_layers.get_weights()[0]
        weights_norm = tf.norm(weights_last_layer, axis=-2)
        weights_norm = tf.squeeze(weights_norm)
        last_bn_layer = activation_extractor.get_layer_input_activation(
            activation_extractor.get_extractor_layer_names()[-1])
        last_bn_layer_norm = tf.norm(tf.reduce_mean(last_bn_layer['input_data'], [1, 2]), axis=-1)

    # Get last linear layer min max
    last_linear_layer_outputs = tf.squeeze(activation_extractor.last_linear_layer_output)

    # In case the last linear layer has more than one output
    if not isinstance(last_linear_layer_outputs, (list, tuple)):
        last_linear_layer_outputs = [last_linear_layer_outputs]

    output_loss = tf.zeros(1)

    for last_linear_layer_output in last_linear_layer_outputs:
        last_layer_out_max = tf.reduce_max(last_linear_layer_output, axis=-1)
        last_layer_out_argmax = tf.argmax(last_linear_layer_output, axis=-1)

        last_layer_out_min = tf.reduce_min(last_linear_layer_output, axis=-1)
        last_layer_out_argmin = tf.argmin(last_linear_layer_output, axis=-1)

        # Add regularization for min max value
        last_layer_reg_min = tf.abs(tf.abs(last_layer_out_min) -
                                    last_bn_layer_norm * tf.gather(weights_norm, last_layer_out_argmin))
        last_layer_reg_max = tf.abs(tf.abs(last_layer_out_max) -
                                    last_bn_layer_norm * tf.gather(weights_norm, last_layer_out_argmax))
        last_layer_dynamic_loss = 1 / (last_layer_out_max - last_layer_out_min + eps)

        output_loss += tf.reduce_mean(last_layer_reg_min + last_layer_reg_max + last_layer_dynamic_loss)

    return output_loss


def min_max_diff(
        output_imgs: tf.Tensor,
        eps: float = 1e-6,
        **kwargs) -> tf.Tensor:
    """
    Calculate the minimum-maximum difference of output images.

    Args:
        output_imgs (Tensor or List[Tensor]): The output of the model on images.
        eps (float): Small value for numerical stability.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The computed minimum-maximum difference loss.
    """
    if not isinstance(output_imgs, (list, tuple)):
        output_imgs = [output_imgs]
    output_loss = 0
    for output in output_imgs:
        output = tf.reshape(output, [output.shape[0], -1])
        output_loss += 1 / (tf.reduce_max(output, 1) - tf.reduce_min(output, 1) + eps)
    return output_loss


def no_output_loss(
        output_imgs: tf.Tensor,
        **kwargs) -> tf.Tensor:
    """
    Calculate no output loss.

    Args:
        output_imgs (Tensor): The output of the model on images.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: A tensor with zero value for the loss.
    """
    return tf.zeros(1)


# Dictionary of output loss functions
output_loss_function_dict: Dict[OutputLossType, Callable] = {
    OutputLossType.NONE: no_output_loss,
    OutputLossType.MIN_MAX_DIFF: min_max_diff,
    OutputLossType.REGULARIZED_MIN_MAX_DIFF: regularized_min_max_diff,
}

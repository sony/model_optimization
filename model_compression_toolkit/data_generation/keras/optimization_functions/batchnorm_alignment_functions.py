from typing import Dict, Callable

import tensorflow as tf

from model_compression_toolkit.data_generation.common.enums import BatchNormAlignemntLossType


def l2_square(bn_mean: tf.Tensor,
              input_mean: tf.Tensor,
              bn_std: tf.Tensor,
              input_std: tf.Tensor) -> tf.Tensor:
    """
    Compute the L2 Square loss for batch normalization alignment.

    Args:
        bn_mean (Tensor): The mean of the batch normalization layer from the original statistics.
        input_mean (Tensor): The mean of the batch normalization layer from the current batch statistics.
        bn_std (Tensor): The standard deviation of the batch normalization layer from the original statistics.
        input_std (Tensor): The standard deviation of the batch normalization layer from the current batch statistics.

    Returns:
        Tensor: The L2 Square loss value for batch normalization alignment.
    """
    return tf.norm(input_mean - bn_mean, axis=-1) ** 2 / bn_mean.shape[0] + \
        tf.norm(input_std - bn_std, axis=-1) ** 2 / bn_std.shape[0]


# Dictionary of batch normalization alignment loss functions
bn_alignment_loss_function_dict: Dict[BatchNormAlignemntLossType, Callable] = {
    BatchNormAlignemntLossType.L2_SQUARE: l2_square,
}

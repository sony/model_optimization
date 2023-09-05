from typing import Dict, Callable

from model_compression_toolkit.data_generation.common.enums import BNLayerWeightingType
from model_compression_toolkit.data_generation.keras.model_info_exctractors import KerasOriginalBNStatsHolder


def average_layer_weighting_fn(orig_bn_stats_holder: KerasOriginalBNStatsHolder, **kwargs) -> Dict[str, float]:
    """
    Calculate average weighting for each batch normalization layer.

    Args:
        orig_bn_stats_holder (KerasOriginalBNStatsHolder): Holder for original batch normalization statistics.
        **kwargs: Additional arguments if needed.

    Returns:
        Dict[str, float]: A dictionary containing layer names as keys and average weightings as values.
    """
    num_bn_layers = orig_bn_stats_holder.get_num_bn_layers()
    return {bn_layer_name: 1 / num_bn_layers for bn_layer_name in orig_bn_stats_holder.get_bn_layer_names()}


def first_bn_multiplier_weighting_fn(orig_bn_stats_holder: KerasOriginalBNStatsHolder, **kwargs) -> Dict[str, float]:
    """
    Calculate layer weightings with a higher multiplier for the first batch normalization layer.

    Args:
        orig_bn_stats_holder (KerasOriginalBNStatsHolder): Holder for original batch normalization statistics.
        **kwargs: Additional arguments if needed.

    Returns:
        Dict[str, float]: A dictionary containing layer names as keys and weightings as values.
    """
    num_bn_layers = orig_bn_stats_holder.get_num_bn_layers()
    layer_weighting_dict = {orig_bn_stats_holder.get_bn_layer_names()[0]: 10 / num_bn_layers}
    layer_weighting_dict.update(
        {bn_layer_name: 1 / num_bn_layers for bn_layer_name in orig_bn_stats_holder.get_bn_layer_names()[1:]})
    return layer_weighting_dict


# Dictionary of layer weighting functions
bn_layer_weighting_function_dict: Dict[BNLayerWeightingType, Callable] = {
    BNLayerWeightingType.AVERAGE: average_layer_weighting_fn,
    BNLayerWeightingType.FIRST_LAYER_MULTIPLIER: first_bn_multiplier_weighting_fn
}

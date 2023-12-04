from model_compression_toolkit.core.common.pruning import ImportanceMetric
from model_compression_toolkit.core.common.pruning.importance_metrics.base_importance_metric import BaseImportanceMetric
from model_compression_toolkit.core.common.pruning.importance_metrics.lfh_importance_metric import LFHImportanceMetric

# A dictionary mapping each importance metric enum to its corresponding class.
im_dict = {ImportanceMetric.LFH: LFHImportanceMetric}

def get_importance_metric(im: ImportanceMetric, **kwargs) -> BaseImportanceMetric:
    """
    Retrieves an instance of the importance metric class based on the specified importance metric enum.

    Args:
        im (ImportanceMetric): An enum value representing the desired importance metric.
        **kwargs: Additional keyword arguments to be passed to the constructor of the importance metric class.

    Returns:
        BaseImportanceMetric: An instance of a class derived from BaseImportanceMetric corresponding to the provided enum.
    """
    # Retrieve the corresponding class for the provided importance metric enum from the dictionary.
    im = im_dict.get(im)

    # Create and return an instance of the importance metric class with the provided keyword arguments.
    return im(**kwargs)


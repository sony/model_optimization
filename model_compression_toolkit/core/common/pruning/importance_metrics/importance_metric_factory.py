from model_compression_toolkit.core.common.pruning import ImportanceMetric
from model_compression_toolkit.core.common.pruning.importance_metrics.base_importance_metric import BaseImportanceMetric
from model_compression_toolkit.core.common.pruning.importance_metrics.lfh_importance_metric import LFHImportanceMetric

im_dict = {ImportanceMetric.LFH: LFHImportanceMetric}


def get_importance_metric(im: ImportanceMetric, **kwargs) -> BaseImportanceMetric:
    im = im_dict.get(im)
    return im(**kwargs)

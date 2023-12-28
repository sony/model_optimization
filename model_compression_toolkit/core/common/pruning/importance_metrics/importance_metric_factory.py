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

from model_compression_toolkit.core.common.pruning.pruning_config import ImportanceMetric
from model_compression_toolkit.core.common.pruning.importance_metrics.base_importance_metric import BaseImportanceMetric
from model_compression_toolkit.core.common.pruning.importance_metrics.lfh_importance_metric import LFHImportanceMetric

# A dictionary mapping each importance metric enum to its corresponding class.
IMPORTANCE_METRIC_DICT = {ImportanceMetric.LFH: LFHImportanceMetric}

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
    im = IMPORTANCE_METRIC_DICT.get(im)

    # Create and return an instance of the importance metric class with the provided keyword arguments.
    return im(**kwargs)


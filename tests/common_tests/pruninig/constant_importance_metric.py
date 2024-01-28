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

from enum import Enum

from typing import List

from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.common.pruning.importance_metrics.base_importance_metric import BaseImportanceMetric
import numpy as np

from model_compression_toolkit.core.common.pruning.importance_metrics.importance_metric_factory import IMPORTANCE_METRIC_DICT


class ConstantImportanceMetric(BaseImportanceMetric):
    """
    ConstantImportanceMetric is used for testing architectures with three linear layers in a row.
    It assigns scores in reverse order of the channel index. It generates constant scores and
    grouped indices for the first two layers based on predefined numbers of output channels.
    """

    # Static attributes to hold the predefined number of output channels for the first two layers.
    first_num_oc = None
    second_num_oc = None
    simd = 1

    def __init__(self, **kwargs):
        pass

    def get_entry_node_to_simd_score(self, entry_nodes: List[BaseNode]):
        """
        Generates the scores and group indices for the provided entry nodes.

        Args:
            entry_nodes (List[BaseNode]): The entry nodes for which scores are to be generated.

        Returns:
            A tuple containing the generated scores and group indices.
        """
        grouped_indices = {
            entry_nodes[0]: [np.arange(i, min(i + ConstantImportanceMetric.simd, ConstantImportanceMetric.first_num_oc)) for i in range(0, ConstantImportanceMetric.first_num_oc, ConstantImportanceMetric.simd)],
            entry_nodes[1]: [np.arange(i, min(i + ConstantImportanceMetric.simd, ConstantImportanceMetric.second_num_oc)) for i in range(0, ConstantImportanceMetric.second_num_oc, ConstantImportanceMetric.simd)]
        }

        entry_node_to_simd_score = {
            entry_nodes[0]: [-np.min(np.arange(i, min(i + ConstantImportanceMetric.simd, ConstantImportanceMetric.first_num_oc))) for i in range(0, ConstantImportanceMetric.first_num_oc, ConstantImportanceMetric.simd)],
            entry_nodes[1]: [-np.min(np.arange(i, min(i + ConstantImportanceMetric.simd, ConstantImportanceMetric.second_num_oc))) for i in range(0, ConstantImportanceMetric.second_num_oc, ConstantImportanceMetric.simd)]
        }

        return entry_node_to_simd_score, grouped_indices


class ConstImportanceMetric(Enum):
    CONST = 'const'


def add_const_importance_metric(first_num_oc, second_num_oc, simd=1):
    """
    Adds the constant importance metric to the global importance metrics dictionary.

    Args:
        first_num_oc (int): Number of output channels for the first layer.
        second_num_oc (int): Number of output channels for the second layer.
    """
    # Set the static attributes for the number of output channels.
    ConstantImportanceMetric.first_num_oc = first_num_oc
    ConstantImportanceMetric.second_num_oc = second_num_oc
    ConstantImportanceMetric.simd = simd

    # Update the global dictionary mapping importance metrics to their corresponding classes.
    IMPORTANCE_METRIC_DICT.update({ConstImportanceMetric.CONST: ConstantImportanceMetric})

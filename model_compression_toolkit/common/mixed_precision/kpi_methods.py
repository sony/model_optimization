# Copyright 2022 Sony Semiconductors Israel, Inc. All rights reserved.
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
from functools import partial

import numpy as np

from model_compression_toolkit.common.mixed_precision.kpi_data import compute_weights_size_kpi, \
    compute_activation_output_size_kpi


def weights_size_kpi(mp_cfg, graph, fw_info):
    """
    Computes a KPIs vector with the respective weights' memory size for each weigh configurable node,
    according to the given mixed-precision configuration.
    Note that the configuration includes an index for each configurable node! (not just weights configurable).

    Args:
        mp_cfg: A mixed-precision configuration (list of candidates index for each configurable node)
        graph: Graph object.
        fw_info: FrameworkInfo object about the specific framework (e.g., attributes of different layers' weights to quantize).

    Returns: A vector of node's weights memory sizes.
    Note that the vector is not necessarily of the same length as the given config.

    """
    return compute_weights_size_kpi(mp_cfg, graph, fw_info)


def activation_output_size_kpi(mp_cfg, graph, fw_info):
    """
    Computes a KPIs vector with the respective output memory size for each activation configurable node,
    according to the given mixed-precision configuration.
    Note that the configuration includes an index for each configurable node! (not just activation configurable).

    Args:
        mp_cfg: A mixed-precision configuration (list of candidates index for each configurable node)
        graph: Graph object.
        fw_info: FrameworkInfo object about the specific framework (e.g., attributes of different layers' weights to quantize)
            (not used in this method).

    Returns: A vector of node's weights memory sizes.
    Note that the vector is not necessarily of the same length as the given config.

    """

    return compute_activation_output_size_kpi(mp_cfg, graph)


class MpKpiMetric(Enum):
    """
    Defines kpi computation functions that can be used to compute KPI for a given target for a given mp config.
    The enum values can be used to call a function on a set of arguments.

     WEIGHTS_SIZE - applies the weights_size_kpi function

     ACTIVATION_OUTPUT_SIZE - applies the activation_output_size_kpi function

    """

    WEIGHTS_SIZE = partial(weights_size_kpi)
    ACTIVATION_OUTPUT_SIZE = partial(activation_output_size_kpi)

    def __call__(self, *args):
        return self.value(*args)

# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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
from typing import NamedTuple

from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization import RUTarget
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.ru_aggregation_methods import MpRuAggregation
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.ru_methods import MpRuMetric


# When adding a RUTarget that we want to consider in our mp search,
# a matching pair of resource_utilization_tools computation function and a resource_utilization_tools
# aggregation function should be added to this dictionary
class RuFunctions(NamedTuple):
    metric_fn: MpRuMetric
    aggregate_fn: MpRuAggregation


ru_functions_mapping = {RUTarget.WEIGHTS: RuFunctions(MpRuMetric.WEIGHTS_SIZE, MpRuAggregation.SUM),
                        RUTarget.ACTIVATION: RuFunctions(MpRuMetric.ACTIVATION_MAXCUT_SIZE, MpRuAggregation.MAX),
                        RUTarget.TOTAL: RuFunctions(MpRuMetric.TOTAL_WEIGHTS_ACTIVATION_SIZE, MpRuAggregation.TOTAL),
                        RUTarget.BOPS: RuFunctions(MpRuMetric.BOPS_COUNT, MpRuAggregation.SUM)}

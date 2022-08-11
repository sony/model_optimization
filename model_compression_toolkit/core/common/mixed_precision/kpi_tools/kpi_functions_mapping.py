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
from model_compression_toolkit.core.common.mixed_precision.kpi_tools.kpi import KPITarget
from model_compression_toolkit.core.common.mixed_precision.kpi_tools.kpi_aggregation_methods import MpKpiAggregation
from model_compression_toolkit.core.common.mixed_precision.kpi_tools.kpi_methods import MpKpiMetric


# When adding a KPITarget that we want to consider in our mp search,
# a matching pair of kpi_tools computation function and a kpi_tools
# aggregation function should be added to this dictionary
kpi_functions_mapping = {KPITarget.WEIGHTS: (MpKpiMetric.WEIGHTS_SIZE, MpKpiAggregation.SUM),
                         KPITarget.ACTIVATION: (MpKpiMetric.ACTIVATION_OUTPUT_SIZE, MpKpiAggregation.MAX),
                         KPITarget.TOTAL: (MpKpiMetric.TOTAL_WEIGHTS_ACTIVATION_SIZE, MpKpiAggregation.TOTAL),
                         KPITarget.BOPS: (MpKpiMetric.BOPS_COUNT, MpKpiAggregation.SUM)}

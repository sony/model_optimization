#  Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==============================================================================

# Default similarity metric names:
CS_SIMILARITY_METRIC_NAME = 'cs'
SQNR_SIMILARITY_METRIC_NAME = 'sqnr'
MSE_SIMILARITY_METRIC_NAME = 'mse'

# Report components names:
OUTPUT_SIMILARITY_METRICS_REPR = 'output_similarity_metrics_repr'
OUTPUT_SIMILARITY_METRICS_VAL = 'output_similarity_metrics_val'
INTERMEDIATE_SIMILARITY_METRICS_REPR = 'intermediate_similarity_metrics_repr'
INTERMEDIATE_SIMILARITY_METRICS_VAL = 'intermediate_similarity_metrics_val'

# Graph attribute names:
XQUANT_REPR = 'xquant_repr'
XQUANT_VAL = 'xquant_val'

# Report file name:
REPORT_FILENAME = 'quant_report.json'

# Tag to use in tensorboard for the graph we plot:
TENSORBOARD_DEFAULT_TAG = 'xquant'

# When extracting the activations of a model we hold the output using a dedicated key:
MODEL_OUTPUT_KEY = 'model_output_key'

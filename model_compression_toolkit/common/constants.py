# Copyright 2021 Sony Semiconductors Israel, Inc. All rights reserved.
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


# Minimal threshold to use for quantization ranges:
MIN_THRESHOLD = (2 ** -28)
EPS = 1e-8
MULTIPLIER_N_BITS = 8

# Quantization attributes:
OUTPUT_SCALE = 'output_scale'
THRESHOLD = 'threshold'
CLUSTER_CENTERS = 'cluster_centers'
SCALE_PER_CHANNEL = 'scale_per_channel'


# Data types:
DATA_TYPE = 'dtype'
FLOAT_32 = 'float32'


# Number of Tensorboard cosine-similarity plots to add:
NUM_SAMPLES_CS_TENSORBOARD = 20


# In Mixed-Precision, a node can have multiple candidates for weights quantization configuration.
# In order to display a single view of a node (for example, for logging in TensorBoard) we need to track the attributes
# that are shared among different candidates:
WEIGHTS_NBITS_ATTRIBUTE = 'weights_n_bits'
CORRECTED_BIAS_ATTRIBUTE = 'corrected_bias'



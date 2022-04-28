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

# Supported frameworks in MCT:
TENSORFLOW = 'tensorflow'
PYTORCH = 'pytorch'

# Minimal threshold to use for quantization ranges:
MIN_THRESHOLD = (2 ** -28)
EPS = 1e-8
MULTIPLIER_N_BITS = 8

# Quantization attributes:
OUTPUT_SCALE = 'output_scale'
THRESHOLD = 'threshold'
SIGNED = 'is_signed'
CLUSTER_CENTERS = 'cluster_centers'
SCALE_PER_CHANNEL = 'scale_per_channel'
RANGE_MIN = 'range_min'
RANGE_MAX = 'range_max'

# BaseNode attributes
REUSE = 'reuse'
REUSE_GROUP = 'reuse_group'

# Data types:
DATA_TYPE = 'dtype'
FLOAT_32 = 'float32'


# Number of Tensorboard cosine-similarity plots to add:
NUM_SAMPLES_CS_TENSORBOARD = 20

# num bits for shift negative non linear node
SHIFT_NEGATIVE_NON_LINEAR_NUM_BITS = 16

# Default bitwidth for disabled quantization candidate
DEFAULT_CANDIDATE_BITWIDTH = 16

# In Mixed-Precision, a node can have multiple candidates for weights and activations quantization configuration.
# In order to display a single view of a node (for example, for logging in TensorBoard) we need to track the attributes
# that are shared among different candidates:
WEIGHTS_NBITS_ATTRIBUTE = 'weights_n_bits'
CORRECTED_BIAS_ATTRIBUTE = 'corrected_bias'
ACTIVATION_NBITS_ATTRIBUTE = 'activation_n_bits'


# Quantization Parameters Iterative Search Defaults:
SYMMETRIC_TENSOR_N_ITER = 40
SYMMETRIC_TENSOR_PER_CHANNEL_N_ITER = 15
SYMMETRIC_HISTOGRAM_N_ITER = 20

UNIFORM_TENSOR_N_ITER = 30
UNIFORM_TENSOR_PER_CHANNEL_N_ITER = 10
UNIFORM_HISTOGRAM_N_ITER = 30

SYMMETRIC_TENSOR_N_INTERVALS = 30
SYMMETRIC_TENSOR_PER_CHANNEL_N_INTERVALS = 30
SYMMETRIC_HISTOGRAM_N_INTERVALS = 30

SYMMETRIC_TENSOR_DEC_FREQ = 5
SYMMETRIC_TENSOR_PER_CHANNEL_DEC_FREQ = 3
SYMMETRIC_HISTOGRAM_DEC_FREQ = 4

UNIFORM_TENSOR_N_SAMPLES = 8
UNIFORM_HISTOGRAM_N_SAMPLES = 12

DEFAULT_DEC_FACTOR = (1.02, 0.98)
DEFAULT_TOL = 1e-11
BOTTOM_FACTOR = 0.7
UPPER_FACTOR = 1.2
DEC_RANGE_BOTTOM = 0.97
DEC_RANGE_UPPER = 1.03

# KPI computation parameters
BITS_TO_BYTES = 8.0

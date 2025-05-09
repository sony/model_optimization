# Copyright 2021 Sony Semiconductor Israel, Inc. All rights reserved.
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


WEIGHTS_SIGNED = True
# Minimal threshold to use for quantization ranges:
MIN_THRESHOLD = (2 ** -16)
EPS = 1e-8
LUT_VALUES_BITWIDTH = 8
FP32_BYTES_PER_PARAMETER = 4.

# Quantization attributes:
OUTPUT_SCALE = 'output_scale'
THRESHOLD = 'threshold'
SIGNED = 'is_signed'
LUT_VALUES = 'lut_values'
SCALE_PER_CHANNEL = 'scale_per_channel'
RANGE_MIN = 'range_min'
RANGE_MAX = 'range_max'

# BaseNode attributes
REUSE = 'reuse'
REUSE_GROUP = 'reuse_group'
LAST_AXIS = -1
AXIS = 'axis'

# Data types:
DATA_TYPE = 'dtype'
FLOAT_32 = 'float32'

# Number of Tensorboard cosine-similarity plots to add:
NUM_SAMPLES_DISTANCE_TENSORBOARD = 20

# num bits for shift negative non linear node
SHIFT_NEGATIVE_NON_LINEAR_NUM_BITS = 16

# Default bitwidth for disabled quantization candidate
FLOAT_BITWIDTH = 32

# In Mixed-Precision, a node can have multiple candidates for weights and activations quantization configuration.
# In order to display a single view of a node (for example, for logging in TensorBoard) we need to track the attributes
# that are shared among different candidates:
WEIGHTS_ATTRIBUTE = 'weights'
ACTIVATION_ATTRIBUTE = 'activation'
WEIGHTS_NBITS_ATTRIBUTE = 'weights_n_bits'
CORRECTED_BIAS_ATTRIBUTE = 'corrected_bias'
ACTIVATION_N_BITS_ATTRIBUTE = 'activation_n_bits'
SUPPORTED_INPUT_ACTIVATION_NBITS_ATTRIBUTE = 'supported_input_activation_n_bits'
WEIGHTS = 'weights'
ACTIVATION = 'activation'

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

NUM_QPARAM_HESSIAN_SAMPLES = 16

# Resource utilization computation parameters
BITS_TO_BYTES = 8.0

# Default threshold for Softmax layer
SOFTMAX_THRESHOLD = 1

# Substitutions node names
VIRTUAL_WEIGHTS_SUFFIX = '_v_weights'
VIRTUAL_ACTIVATION_SUFFIX = '_v_activation'
VIRTUAL_ACTIVATION_WEIGHTS_NODE_PREFIX = 'virtual'

# Memory graph constants
DUMMY_NODE = 'dummy_node'
DUMMY_TENSOR = 'dummy_tensor'

# Hessian scores constants
MIN_HESSIAN_ITER = 10
HESSIAN_COMP_TOLERANCE = 1e-3


# Hessian configuration default constants
HESSIAN_OUTPUT_ALPHA = 0.3
HESSIAN_NUM_ITERATIONS = 100
HESSIAN_EPS = 1e-6
ACT_HESSIAN_DEFAULT_BATCH_SIZE = 32
GPTQ_HESSIAN_NUM_SAMPLES = 32
MP_DEFAULT_NUM_SAMPLES = 32

# Pruning constants
PRUNING_NUM_SCORE_APPROXIMATIONS = 32

# Scheduling information fields
OPERATORS_SCHEDULING = 'operators_scheduling'
MAX_CUT = 'max_cut'
CUTS = 'cuts'
FUSED_NODES_MAPPING = 'fused_nodes_mapping'
OP_ORDER = 'op_order'
OP_RECORD = 'op_record'
MEM_ELEMENTS = 'mem_elements'
SHAPE = 'shape'
NODE_NAME = 'node_name'
TOTAL_SIZE = 'total_size'
NODE_OUTPUT_INDEX = 'node_output_index'


# Fusing Patterns constants
FUSED_LAYER_PATTERN = 'fused_layer_pattern'
FUSED_OP_QUANT_CONFIG = 'fused_op_quantization_config'
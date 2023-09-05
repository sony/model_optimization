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

# Define a constant for the image input key.
IMAGE_INPUT = 'image_input'

# Define constants for image axes.
BATCH_AXIS, H_AXIS, W_AXIS, CHANNEL_AXIS = 0, 1, 2, 3

# Default initial learning rate constant.
DEFAULT_INITIAL_LR = 1

# Default batch size for data generator.
DEFAULT_DATA_GEN_BS = 8

# Default number of iterations.
DEFAULT_N_ITER = 500

# Default output loss multiplier.
DEFAULT_OUTPUT_LOSS_MULTIPLIER = 0.001

# Minimum value for image pixel intensity.
IMAGE_MIN_VAL = 0.0

# Maximum value for image pixel intensity.
IMAGE_MAX_VAL = 255.0


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
import torch

# Axis indices for tensor dimensions
BATCH_AXIS, CHANNEL_AXIS, H_AXIS, W_AXIS = 0, 1, 2, 3

# Default initial learning rate constant.
DEFAULT_PYTORCH_INITIAL_LR = 16

# Default extra pixels for image padding.
DEFAULT_PYTORCH_EXTRA_PIXELS = 32

# Default output loss multiplier.
DEFAULT_PYTORCH_OUTPUT_LOSS_MULTIPLIER = 1e-5

# Default BatchNorm layer types
DEFAULT_PYTORCH_BN_LAYER_TYPES = [torch.nn.BatchNorm2d]

# Default last layer types
DEFAULT_PYTORCH_LAST_LAYER_TYPES = [torch.nn.Linear, torch.nn.Conv2d]

# Output string
OUTPUT = 'output'
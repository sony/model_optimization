# Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
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


class TrainingMethod(Enum):
    """
    An enum for selecting a training method

    STE - Standard straight-through estimator. Includes PowerOfTwo, symmetric & uniform quantizers

    DQA -  DNN Quantization with Attention. Includes a smooth quantization introduces by DQA method

    LSQ - Learned Step size Quantization. Includes PowerOfTwo, symmetric & uniform quantizers: https://arxiv.org/pdf/1902.08153.pdf

    """
    STE = "STE",
    DQA = "DQA",
    LSQ = "LSQ"

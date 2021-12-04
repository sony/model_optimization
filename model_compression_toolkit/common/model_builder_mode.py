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

from enum import Enum


class ModelBuilderMode(Enum):
    """
    Mode for building the model back from a graph:
    FLOAT - Build model for statistics collection. Model's outputs list contain all output tensors of all nodes
    in the graph.
    QUANTIZED - Build a quantized model using the nodes' quantization attributes for adding
    quantization nodes to the model.
    GPTQ - Build a quantized model using the nodes' quantization attributes for wrapping
    layers with QuantizeWrapper and output comparing points.
    MIXEDPRECISION - Build a quantized model where the layers that their weights should be quantized
    can use different quantized weights according to the possible bitwidths of each layer.
    """
    FLOAT = 0
    QUANTIZED = 1
    GPTQ = 2
    MIXEDPRECISION = 3
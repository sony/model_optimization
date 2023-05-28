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
import keras

from model_compression_toolkit.quantizers_infrastructure import KerasQuantizationWrapper


def get_layers_from_model_by_type(model:keras.Model,
                                  layer_type: type,
                                  include_wrapped_layers: bool = True):
    if include_wrapped_layers:
        return [layer for layer in model.layers if type(layer)==layer_type or (isinstance(layer, KerasQuantizationWrapper) and type(layer.layer)==layer_type)]
    return [layer for layer in model.layers if type(layer)==layer_type]






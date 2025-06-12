# Copyright 2025 Sony Semiconductor Israel, Inc. All rights reserved.
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
from typing import List, Tuple

import numpy as np

from model_compression_toolkit.core.common.framework_info import set_fw_info
from model_compression_toolkit.core.keras.default_framework_info import KerasInfo
from model_compression_toolkit.core.keras.keras_implementation import KerasImplementation
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.attach2keras import \
    AttachTpcToKeras


class KerasFwMixin:
    """ Mixin helper class containing keras-specific definitions. """
    set_fw_info(KerasInfo)
    fw_impl = KerasImplementation()
    attach_to_fw_func = AttachTpcToKeras().attach

    def get_basic_data_gen(self, shapes: List[Tuple]):
        """ Generate a basic data generator. """
        def f():
            yield [np.random.randn(*shape).astype(np.float32) for shape in shapes]
        return f

    @staticmethod
    def fetch_model_layers_by_cls(model, cls):
        """ Fetch layers from keras model by layer class type. """
        return [m for m in model.layers if isinstance(m, cls)]

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
from model_compression_toolkit.common.constants import TENSORFLOW
from model_compression_toolkit.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.keras.constants import DEFAULT_TP_MODEL
from model_compression_toolkit.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit import keras_post_training_quantization, \
    keras_post_training_quantization_mixed_precision, FrameworkInfo
from model_compression_toolkit import get_model
from model_compression_toolkit.keras.keras_implementation import KerasImplementation
from tests.common_tests.base_feature_test import BaseFeatureNetworkTest


class BaseKerasFeatureNetworkTest(BaseFeatureNetworkTest):
    def __init__(self,
                 unit_test,
                 num_calibration_iter=1,
                 val_batch_size=1,
                 num_of_inputs=1,
                 input_shape=(8, 8, 3)):

        super().__init__(unit_test=unit_test,
                         val_batch_size=val_batch_size,
                         num_calibration_iter=num_calibration_iter,
                         num_of_inputs=num_of_inputs,
                         input_shape=input_shape)

    def get_fw_hw_model(self):
        return get_model(TENSORFLOW, DEFAULT_TP_MODEL)

    def get_ptq_facade(self):
        return keras_post_training_quantization

    def get_mixed_precision_ptq_facade(self):
        return keras_post_training_quantization_mixed_precision

    def get_fw_info(self) -> FrameworkInfo:
        return DEFAULT_KERAS_INFO

    def get_fw_impl(self) -> FrameworkImplementation:
        return KerasImplementation()



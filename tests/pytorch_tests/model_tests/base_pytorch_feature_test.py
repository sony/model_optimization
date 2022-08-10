# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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
from model_compression_toolkit.core.common.constants import PYTORCH
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.pytorch.constants import DEFAULT_TP_MODEL
from model_compression_toolkit import pytorch_post_training_quantization, pytorch_post_training_quantization_mixed_precision, FrameworkInfo
from model_compression_toolkit import get_target_platform_capabilities
from model_compression_toolkit.core.pytorch.pytorch_implementation import PytorchImplementation
from model_compression_toolkit.core.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO
from tests.common_tests.base_feature_test import BaseFeatureNetworkTest


class BasePytorchFeatureNetworkTest(BaseFeatureNetworkTest):
    def __init__(self,
                 unit_test,
                 num_calibration_iter=1,
                 val_batch_size=1,
                 num_of_inputs=1,
                 input_shape=(3, 8, 8)):

        super().__init__(unit_test=unit_test,
                         val_batch_size=val_batch_size,
                         num_calibration_iter=num_calibration_iter,
                         num_of_inputs=num_of_inputs,
                         input_shape=input_shape)

    def get_tpc(self):
        return get_target_platform_capabilities(PYTORCH, DEFAULT_TP_MODEL)

    def get_ptq_facade(self):
        return pytorch_post_training_quantization

    def get_mixed_precision_ptq_facade(self):
        return pytorch_post_training_quantization_mixed_precision

    def get_fw_info(self) -> FrameworkInfo:
        return DEFAULT_PYTORCH_INFO

    def get_fw_impl(self) -> FrameworkImplementation:
        return PytorchImplementation()



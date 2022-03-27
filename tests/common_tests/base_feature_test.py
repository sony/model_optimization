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
from typing import List

import numpy as np

from model_compression_toolkit.common.mixed_precision.mixed_precision_quantization_config import \
    MixedPrecisionQuantizationConfig
from model_compression_toolkit.common.user_info import UserInformation
from tests.common_tests.base_test import BaseTest
from tests.common_tests.base_layer_test import LayerTestMode


class BaseFeatureNetworkTest(BaseTest):
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

    def get_gptq_config(self):
        return None

    def get_network_editor(self):
        return []

    def get_kpi(self):
        return None

    def run_test(self):
        feature_networks = self.create_networks()
        feature_networks = feature_networks if isinstance(feature_networks, list) else [feature_networks]
        for model_float in feature_networks:
            qc = self.get_quantization_config()
            if isinstance(qc, MixedPrecisionQuantizationConfig):
                ptq_model, quantization_info = self.get_mixed_precision_ptq_facade()(model_float,
                                                                                     self.representative_data_gen,
                                                                                     n_iter=self.num_calibration_iter,
                                                                                     quant_config=qc,
                                                                                     fw_info=self.get_fw_info(),
                                                                                     network_editor=self.get_network_editor(),
                                                                                     gptq_config=self.get_gptq_config(),
                                                                                     target_kpi=self.get_kpi(),
                                                                                     fw_hw_model=self.get_fw_hw_model())
            else:
                ptq_model, quantization_info = self.get_ptq_facade()(model_float,
                                                                     self.representative_data_gen,
                                                                     n_iter=self.num_calibration_iter,
                                                                     quant_config=qc,
                                                                     fw_info=self.get_fw_info(),
                                                                     network_editor=self.get_network_editor(),
                                                                     gptq_config=self.get_gptq_config(),
                                                                     fw_hw_model=self.get_fw_hw_model())

            self.compare(ptq_model,
                         model_float,
                         input_x=self.representative_data_gen(),
                         quantization_info=quantization_info)



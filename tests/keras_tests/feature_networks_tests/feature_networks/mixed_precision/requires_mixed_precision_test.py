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

import numpy as np

from model_compression_toolkit.core import ResourceUtilization
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization_data import \
    compute_resource_utilization_data
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.core.keras.keras_implementation import KerasImplementation
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import \
    get_op_quantization_configs
from tests.keras_tests.feature_networks_tests.feature_networks.mixed_precision_tests import get_base_mp_nbits_candidates
from tests.keras_tests.feature_networks_tests.feature_networks.weights_mixed_precision_tests import \
    MixedPrecisionBaseTest
from tests.keras_tests.tpc_keras import get_tpc_with_activation_mp_keras, get_weights_only_mp_tpc_keras
from tests.common_tests.helpers.generate_test_tp_model import generate_test_op_qc, generate_test_attr_configs
import model_compression_toolkit as mct


class RequiresMixedPrecision(MixedPrecisionBaseTest):
    """
    A test to ensure that mixed precision is used instead of single precision
    when the memory allocated for resources is less than the memory required for all layers using single precision.
    """
    def __init__(self, unit_test, weights_memory=False, activation_memory=False, bops=False, total_memory=False):
        super().__init__(unit_test, val_batch_size=1)
        self.weights_memory = weights_memory
        self.activation_memory = activation_memory
        self.bops = bops
        self.total_memory = total_memory

    def get_tpc(self):
        eight_bits = generate_test_op_qc(**generate_test_attr_configs())

        # sets all combinations of 2, 4, 8 bits for weights and activations
        mixed_precision_candidates_list = get_base_mp_nbits_candidates()

        return get_tpc_with_activation_mp_keras(base_config=eight_bits,
                                                default_config=eight_bits,
                                                mp_bitwidth_candidates_list=mixed_precision_candidates_list,
                                                name="")

    def get_max_resources_for_model(self, model):
        return compute_resource_utilization_data(in_model=model,
                                                 representative_data_gen=self.representative_data_gen(),
                                                 core_config=self.get_core_config(),
                                                 tpc=self.get_tpc(),
                                                 fw_info=DEFAULT_KERAS_INFO,
                                                 fw_impl=KerasImplementation(),
                                                 transformed_graph=None,
                                                 mixed_precision_enable=False)

    def get_quantization_config(self):
        return mct.core.QuantizationConfig(mct.core.QuantizationErrorMethod.MSE,
                                           mct.core.QuantizationErrorMethod.MSE,
                                           relu_bound_to_power_of_2=True,
                                           weights_bias_correction=True,
                                           input_scaling=False,
                                           activation_channel_equalization=True)

    def get_resource_utilization(self):
        ru_data = self.get_max_resources_for_model(self.create_networks())
        return ResourceUtilization(weights_memory=ru_data.weights_memory - 1 if self.weights_memory else np.inf,
                                   activation_memory=ru_data.activation_memory - 1 if self.activation_memory else np.inf,
                                   total_memory=ru_data.total_memory - 1 if self.total_memory else np.inf,
                                   bops=int(ru_data.bops * 0.05) if self.bops else np.inf)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        if self.weights_memory or self.activation_memory or self.total_memory or self.bops:
            self.unit_test.assertTrue(any([i != 0 for i in quantization_info.mixed_precision_cfg]))
        else:
            self.unit_test.assertTrue(len(quantization_info.mixed_precision_cfg) == 0)


class RequiresMixedPrecisionWeights(RequiresMixedPrecision):
    def get_tpc(self):
        base_config, _, default_config = get_op_quantization_configs()

        return get_weights_only_mp_tpc_keras(base_config=base_config,
                                             default_config=default_config,
                                             mp_bitwidth_candidates_list=[(8, 8), (2, 8), (3, 8)],
                                             name="mp_test")
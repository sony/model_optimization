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


import numpy as np
import tensorflow as tf

from mct_quantizers import KerasQuantizationWrapper, KerasActivationQuantizationHolder
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import get_op_quantization_configs

from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization import \
    ResourceUtilization
from tests.keras_tests.feature_networks_tests.feature_networks.weights_mixed_precision_tests import \
    MixedPrecisionBaseTest
from tests.keras_tests.utils import get_layers_from_model_by_type

keras = tf.keras
layers = keras.layers


class MixedPrecisionWeightsNoResourceUtilizationForActivationsTest(MixedPrecisionBaseTest):
    """
    This test verifies that if mixed precision is relevant for weights only (namely, it was configured in the
    RU to limit weights only) - the activation bit-widths are as the base config, which means it did not
    search bit-widths for activations during mixed precision.
    """
    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(32, 4)(inputs)
        x = layers.Add()([x, x])
        outputs = layers.ReLU()(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model

    def get_tpc(self):
        from tests.common_tests.helpers.tpcs_for_tests.v4.tpc import get_tpc
        return get_tpc()
    def get_resource_utilization(self):
        return ResourceUtilization(weights_memory=400)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        act_q_layers = get_layers_from_model_by_type(quantized_model, KerasActivationQuantizationHolder)
        act_bits = np.array([l.activation_holder_quantizer.num_bits for l in act_q_layers])
        self.unit_test.assertTrue(np.all(act_bits==8))

class MixedPrecisionActivationNoResourceUtilizationForWeightsTest(MixedPrecisionBaseTest):
    """
    This test verifies that if mixed precision is relevant for activations only (namely, it was configured in the
    RU to limit activations only) - the weights bit-widths are as the base config, which means it did not
    search bit-widths for weights during mixed precision.
    """
    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(32, 4)(inputs)
        x = layers.Add()([x, x])
        outputs = layers.ReLU()(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model

    def get_tpc(self):
        """
        Creates candidate for weight quantization using 16 bits to check the MCT does not use it, since
        we configure the target RU with activation limit only.
        """
        from model_compression_toolkit.target_platform_capabilities.constants import KERNEL_ATTR, WEIGHTS_N_BITS
        from tests.common_tests.helpers.tpcs_for_tests.v4.tpc import generate_tpc
        base_config, mixed_precision_cfg_list, default_config = get_op_quantization_configs()
        mixed_precision_cfg_list.append(mixed_precision_cfg_list[0].clone_and_edit(attr_to_edit={KERNEL_ATTR: {WEIGHTS_N_BITS: 16}}))
        return generate_tpc(default_config=default_config,
                            base_config=base_config,
                            mixed_precision_cfg_list=mixed_precision_cfg_list,
                            name='imx500_tpc')
    def get_resource_utilization(self):
        return ResourceUtilization(activation_memory=99999)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        w_q_layers = get_layers_from_model_by_type(quantized_model, KerasQuantizationWrapper)
        w_bits = np.array([l.weights_quantizers['kernel'].num_bits for l in w_q_layers])
        self.unit_test.assertTrue(np.all(w_bits==8))




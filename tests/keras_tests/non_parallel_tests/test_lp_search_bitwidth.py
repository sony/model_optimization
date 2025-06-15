# Copyright 2021 Sony Semiconductor Israel, Inc. All rights reserved.
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
import unittest

import keras
from model_compression_toolkit.core import DEFAULTCONFIG
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization import \
    ResourceUtilization
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_quantization_config import \
    MixedPrecisionQuantizationConfig, MpDistanceWeighting
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_search_facade import search_bit_width, \
    BitWidthSearchMethod
from model_compression_toolkit.core.common.mixed_precision.sensitivity_eval.sensitivity_evaluation import SensitivityEvaluation
from model_compression_toolkit.core.common.model_collector import ModelCollector
from model_compression_toolkit.core.common.quantization.core_config import CoreConfig
from model_compression_toolkit.core.common.quantization.quantization_params_generation.qparams_computation import \
    calculate_quantization_params
from model_compression_toolkit.core.common.quantization.set_node_quantization_config import \
    set_quantization_configuration_to_graph
from model_compression_toolkit.core.common.similarity_analyzer import compute_mse
from model_compression_toolkit.core.keras.keras_implementation import KerasImplementation
from model_compression_toolkit.target_platform_capabilities.constants import KERNEL_ATTR
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.attach2keras import \
    AttachTpcToKeras
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import \
    get_op_quantization_configs
from tests.keras_tests.tpc_keras import get_weights_only_mp_tpc_keras


class TestSearchBitwidthConfiguration(unittest.TestCase):

    def run_search_bitwidth_config_test(self, core_config):
        base_config, mixed_precision_cfg_list, default_config = get_op_quantization_configs()
        base_config = base_config.clone_and_edit(enable_activation_quantization=False)

        tpc = get_weights_only_mp_tpc_keras(base_config=base_config,
                                            default_config=default_config,
                                            mp_bitwidth_candidates_list=[
                                                (c.attr_weights_configs_mapping[KERNEL_ATTR].weights_n_bits,
                                                 c.activation_n_bits) for c
                                                in mixed_precision_cfg_list],
                                            name="bitwidth_cfg_test")

        input_shape = (1, 8, 8, 3)
        input_tensor = keras.layers.Input(shape=input_shape[1:])
        conv = keras.layers.Conv2D(3, 3)(input_tensor)
        bn = keras.layers.BatchNormalization()(conv)
        relu = keras.layers.ReLU()(bn)
        in_model = keras.Model(inputs=input_tensor, outputs=relu)
        keras_impl = KerasImplementation()

        def dummy_representative_dataset():
            return None

        graph = keras_impl.model_reader(in_model, dummy_representative_dataset)  # model reading

        fqc = AttachTpcToKeras().attach(tpc)

        graph.set_fqc(fqc)
        graph = set_quantization_configuration_to_graph(graph=graph,
                                                        quant_config=core_config.quantization_config,
                                                        mixed_precision_enable=True)

        for node in graph.nodes:
            node.prior_info = keras_impl.get_node_prior_info(node=node,
                                                             graph=graph)

        mi = ModelCollector(graph,
                            fw_impl=keras_impl,
                            qc=core_config.quantization_config)

        for i in range(1):
            mi.infer([np.random.randn(*input_shape)])

        def representative_data_gen():
            yield [np.random.random(input_shape)]

        calculate_quantization_params(graph, fw_impl=keras_impl, repr_data_gen_fn=representative_data_gen)

        SensitivityEvaluation(graph, core_config.mixed_precision_config, representative_data_gen,
                              fw_impl=keras_impl)

        cfg = search_bit_width(graph=graph,
                               fw_impl=keras_impl,
                               target_resource_utilization=ResourceUtilization(weights_memory=100),
                               mp_config=core_config.mixed_precision_config,
                               representative_data_gen=representative_data_gen,
                               search_method=BitWidthSearchMethod.INTEGER_PROGRAMMING)

    def test_mixed_precision_search_facade(self):
        core_config_avg_weights = CoreConfig(quantization_config=DEFAULTCONFIG,
                                             mixed_precision_config=MixedPrecisionQuantizationConfig(compute_mse,
                                                                                                     MpDistanceWeighting.AVG,
                                                                                                     num_of_images=1,
                                                                                                     use_hessian_based_scores=False))

        self.run_search_bitwidth_config_test(core_config_avg_weights)

        core_config_last_layer = CoreConfig(quantization_config=DEFAULTCONFIG,
                                            mixed_precision_config=MixedPrecisionQuantizationConfig(compute_mse,
                                                                                                    MpDistanceWeighting.LAST_LAYER,
                                                                                                    num_of_images=1,
                                                                                                    use_hessian_based_scores=False))

        self.run_search_bitwidth_config_test(core_config_last_layer)


if __name__ == '__main__':
    unittest.main()

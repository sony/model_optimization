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
import unittest
from keras.applications.densenet import DenseNet121
from keras.applications.mobilenet_v2 import MobileNetV2

from model_compression_toolkit.core.common.mixed_precision.distance_weighting import get_average_weights
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_quantization_config import \
    MixedPrecisionQuantizationConfig
from model_compression_toolkit.core.common.mixed_precision.sensitivity_evaluation import get_mp_interest_points
from model_compression_toolkit import DEFAULTCONFIG
from model_compression_toolkit.core.common.quantization.set_node_quantization_config import \
    set_quantization_configuration_to_graph
from model_compression_toolkit.core.common.similarity_analyzer import compute_mse
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.core.keras.keras_implementation import KerasImplementation
from model_compression_toolkit.core.tpc_models.default_tpc.latest import get_op_quantization_configs, generate_keras_tpc
from tests.common_tests.helpers.generate_test_tp_model import generate_mixed_precision_test_tp_model


def build_ip_list_for_test(in_model, num_interest_points_factor):
    qc = MixedPrecisionQuantizationConfig(DEFAULTCONFIG,
                                          compute_mse,
                                          get_average_weights,
                                          num_of_images=1,
                                          num_interest_points_factor=num_interest_points_factor)
    fw_info = DEFAULT_KERAS_INFO
    keras_impl = KerasImplementation()

    def dummy_representative_dataset():
        return None

    graph = keras_impl.model_reader(in_model, dummy_representative_dataset)  # model reading
    graph.set_fw_info(fw_info)

    base_config, mixed_precision_cfg_list = get_op_quantization_configs()
    base_config = base_config.clone_and_edit(enable_activation_quantization=False)
    tp_model = generate_mixed_precision_test_tp_model(
        base_cfg=base_config,
        mp_bitwidth_candidates_list=[(c.weights_n_bits, c.activation_n_bits) for c in mixed_precision_cfg_list])
    tpc = generate_keras_tpc(name="sem_test", tp_model=tp_model)
    graph.set_tpc(tpc)
    graph = set_quantization_configuration_to_graph(graph=graph,
                                                    quant_config=qc,
                                                    mixed_precision_enable=True)

    ips = get_mp_interest_points(graph=graph,
                                 interest_points_classifier=keras_impl.count_node_for_mixed_precision_interest_points,
                                 num_ip_factor=qc.num_interest_points_factor)

    return ips, graph, fw_info


class TestSensitivityMetricInterestPoints(unittest.TestCase):

    def test_filtered_interest_points_set(self):
        in_model = DenseNet121()
        ips, graph, fw_info = build_ip_list_for_test(in_model, num_interest_points_factor=0.5)
        sorted_nodes = graph.get_topo_sorted_nodes()
        ip_nodes = list(filter(lambda n: KerasImplementation().count_node_for_mixed_precision_interest_points(n),
                               sorted_nodes))

        self.assertTrue(len(ips) <= 0.5 * len(ip_nodes),
                        f"Filtered interest points list should include not more than {0.5 * len(ip_nodes)}, but it"
                        f" includes {len(ips)}")

    def test_nonfiltered_interest_points_set(self):
        in_model = MobileNetV2()
        ips, graph, fw_info = build_ip_list_for_test(in_model, num_interest_points_factor=1.0)
        sorted_nodes = graph.get_topo_sorted_nodes()
        ip_nodes = list(filter(lambda n: KerasImplementation().count_node_for_mixed_precision_interest_points(n),
                               sorted_nodes))

        self.assertTrue(len(ips) == len(ip_nodes),
                        f"Filtered interest points list should include exactly {len(ip_nodes)}, but it"
                        f"includes {len(ips)}")

    def test_invalid_interest_points_factor(self):
        in_model = MobileNetV2()

        with self.assertRaises(Exception):
            ips, graph, fw_info = build_ip_list_for_test(in_model, num_interest_points_factor=1.1)
        with self.assertRaises(Exception):
            ips, graph, fw_info = build_ip_list_for_test(in_model, num_interest_points_factor=0)




if __name__ == '__main__':
    unittest.main()

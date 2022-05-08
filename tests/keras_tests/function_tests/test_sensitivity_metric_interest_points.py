# Copyright 2022 Sony Semiconductors Israel, Inc. All rights reserved.
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

from model_compression_toolkit.common.constants import NUM_INTEREST_POINTS_BOUND
from model_compression_toolkit.common.mixed_precision.distance_weighting import get_average_weights
from model_compression_toolkit.common.mixed_precision.mixed_precision_quantization_config import \
    MixedPrecisionQuantizationConfig
from model_compression_toolkit.common.mixed_precision.sensitivity_evaluation_manager import get_mp_interest_points
from model_compression_toolkit import DEFAULTCONFIG
from model_compression_toolkit.common.quantization.set_node_quantization_config import \
    set_quantization_configuration_to_graph
from model_compression_toolkit.common.similarity_analyzer import compute_mse
from model_compression_toolkit.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.keras.keras_implementation import KerasImplementation
from model_compression_toolkit.tpc_models.default_tp_model import get_op_quantization_configs
from model_compression_toolkit.tpc_models.keras_tp_models.keras_default import generate_keras_default_tpc
from tests.common_tests.helpers.generate_test_tp_model import generate_mixed_precision_test_tp_model


def build_ip_list_for_test(in_model):
    qc = MixedPrecisionQuantizationConfig(DEFAULTCONFIG,
                                          compute_mse,
                                          get_average_weights,
                                          num_of_images=1)
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
    tpc = generate_keras_default_tpc(name="sem_test", tp_model=tp_model)
    graph.set_tpc(tpc)
    graph = set_quantization_configuration_to_graph(graph=graph,
                                                    quant_config=qc)

    ips = get_mp_interest_points(graph=graph,
                                 fw_info=fw_info)

    return ips, graph, fw_info


class TestSensitivityMetricInterestPOints(unittest.TestCase):

    def test_filtered_interest_points_set(self):
        in_model = DenseNet121()
        ips, graph, fw_info = build_ip_list_for_test(in_model)
        sorted_nodes = graph.get_topo_sorted_nodes()
        kernel_nodes = list(filter(lambda n: fw_info.get_kernel_op_attributes(n.type)[0] is not None, sorted_nodes))

        self.assertTrue(len(ips) <= NUM_INTEREST_POINTS_BOUND,
                        f"Filtered interest points list should include less than {NUM_INTEREST_POINTS_BOUND}, but it"
                        f"includes {len(ips)}")
        self.assertTrue(len(ips) != len(kernel_nodes), f"Model {in_model.name} should originally have more interest "
                                                       f"points than allowed, thus, the returned list is supposed to "
                                                       f"be smaller")

    def test_nonfiltered_interest_points_set(self):
        in_model = MobileNetV2()
        ips, graph, fw_info = build_ip_list_for_test(in_model)
        sorted_nodes = graph.get_topo_sorted_nodes()
        kernel_nodes = list(filter(lambda n: fw_info.get_kernel_op_attributes(n.type)[0] is not None, sorted_nodes))

        self.assertTrue(len(ips) <= NUM_INTEREST_POINTS_BOUND,
                        f"Filtered interest points list should include less than {NUM_INTEREST_POINTS_BOUND}, but it"
                        f"includes {len(ips)}")
        self.assertTrue(len(ips) == len(kernel_nodes),
                        f"Model {in_model.name} should originally have less interest "
                        f"points than the upper bound, thus, the returned list is supposed to "
                        f"be similar")


if __name__ == '__main__':
    unittest.main()

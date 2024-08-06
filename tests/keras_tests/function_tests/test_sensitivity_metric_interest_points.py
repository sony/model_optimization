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
import tensorflow as tf
import unittest
import numpy as np
from keras.applications.densenet import DenseNet121
from keras.applications.mobilenet_v2 import MobileNetV2

from packaging import version

from model_compression_toolkit.target_platform_capabilities.constants import KERNEL_ATTR

if version.parse(tf.__version__) >= version.parse("2.13"):
    from keras.src.layers.core import TFOpLambda
else:
    from keras.layers.core import TFOpLambda

from model_compression_toolkit.constants import AXIS
from model_compression_toolkit.core.common.mixed_precision.distance_weighting import MpDistanceWeighting
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_quantization_config import \
    MixedPrecisionQuantizationConfig
from model_compression_toolkit.core.common.mixed_precision.sensitivity_evaluation import get_mp_interest_points
from model_compression_toolkit.core import DEFAULTCONFIG
from model_compression_toolkit.core.common.quantization.set_node_quantization_config import \
    set_quantization_configuration_to_graph
from model_compression_toolkit.core.common.similarity_analyzer import compute_mse, compute_kl_divergence
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.core.keras.keras_implementation import KerasImplementation
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import get_op_quantization_configs, generate_keras_tpc
from tests.keras_tests.tpc_keras import get_weights_only_mp_tpc_keras


def build_ip_list_for_test(in_model, num_interest_points_factor):
    mp_qc = MixedPrecisionQuantizationConfig(compute_mse,
                                             MpDistanceWeighting.AVG,
                                             num_of_images=1,
                                             num_interest_points_factor=num_interest_points_factor)
    fw_info = DEFAULT_KERAS_INFO
    keras_impl = KerasImplementation()

    def dummy_representative_dataset():
        return None

    graph = keras_impl.model_reader(in_model, dummy_representative_dataset)  # model reading
    graph.set_fw_info(fw_info)

    base_config, mixed_precision_cfg_list, default_config = get_op_quantization_configs()
    base_config = base_config.clone_and_edit(enable_activation_quantization=False)

    tpc = get_weights_only_mp_tpc_keras(base_config=base_config,
                                        default_config=default_config,
                                        mp_bitwidth_candidates_list=[(c.attr_weights_configs_mapping[KERNEL_ATTR].weights_n_bits,
                                                                      c.activation_n_bits) for c in mixed_precision_cfg_list],
                                        name="sem_test")

    graph.set_tpc(tpc)
    graph = set_quantization_configuration_to_graph(graph=graph,
                                                    quant_config=DEFAULTCONFIG,
                                                    mixed_precision_enable=True)

    ips = get_mp_interest_points(graph=graph,
                                 interest_points_classifier=keras_impl.count_node_for_mixed_precision_interest_points,
                                 num_ip_factor=mp_qc.num_interest_points_factor)

    return ips, graph, fw_info


def softmax_model(input_shape):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, 4)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Softmax(axis=2)(x)
    x = tf.keras.layers.Dense(32)(x)
    x = tf.nn.softmax(x, axis=-1)
    outputs = tf.keras.layers.Reshape((-1,))(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model



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

        # Note that the model's output node is shouldn't be included in the sensitivity evaluation list of
        # interest points (it is included in a separate list)
        self.assertTrue(len(ips) == len(ip_nodes) - 1,
                        f"Filtered interest points list should include exactly {len(ip_nodes)}, but it"
                        f"includes {len(ips)}")

    def test_invalid_interest_points_factor(self):
        in_model = MobileNetV2()

        with self.assertRaises(Exception):
            ips, graph, fw_info = build_ip_list_for_test(in_model, num_interest_points_factor=1.1)
        with self.assertRaises(Exception):
            ips, graph, fw_info = build_ip_list_for_test(in_model, num_interest_points_factor=0)

    def test_softmax_interest_point(self):
        in_model = softmax_model((16, 16, 3))
        ips, graph, fw_info = build_ip_list_for_test(in_model, num_interest_points_factor=1.0)

        softmax_nodes = [n for n in graph.get_topo_sorted_nodes() if n.layer_class == tf.keras.layers.Softmax or
                         (n.layer_class == TFOpLambda and n.framework_attr['function'] == 'nn.softmax')]
        softmax_node2layer = {n: [l for l in in_model.layers if isinstance(l, n.layer_class)][0] for n in softmax_nodes}

        self.assertTrue(len(softmax_nodes) == 2)

        for sn in softmax_nodes:
            self.assertIn(sn, ips, f"Expecting a softmax layer to be considered as interest point for "
                                   f"mixed precision distance metric but node {sn.name} is missing.")

            t1 = softmax_node2layer[sn](np.random.rand(*[8, *softmax_node2layer[sn].input_shape[1:]])).numpy()
            t2 = softmax_node2layer[sn](np.random.rand(*[8, *softmax_node2layer[sn].input_shape[1:]])).numpy()

            axis = sn.framework_attr.get(AXIS)
            if axis is None:
                axis = sn.op_call_kwargs.get(AXIS)

            distance_fn, _ = KerasImplementation().get_mp_node_distance_fn(sn)
            self.assertEqual(distance_fn, compute_kl_divergence,
                             f"Softmax node should use KL Divergence for distance computation.")

            distance_per_softmax_axis = distance_fn(t1, t2, batch=True, axis=axis)
            distance_global = distance_fn(t1, t2, batch=True, axis=None)

            self.assertFalse(np.isclose(np.mean(distance_per_softmax_axis), distance_global),
                             f"Computing distance for softmax node on softmax activation axis should be different than "
                             f"on than computing on the entire tensor.")


if __name__ == '__main__':
    unittest.main()

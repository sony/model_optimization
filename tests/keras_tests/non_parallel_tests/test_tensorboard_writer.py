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

import glob
import os
import unittest

import numpy as np
import tensorflow as tf
from tensorboard.backend.event_processing import event_file_loader
from tensorboard.compat.proto.graph_pb2 import GraphDef

import model_compression_toolkit as mct
from model_compression_toolkit.core import QuantizationConfig
from model_compression_toolkit.core.common.visualization.final_config_visualizer import \
    ActivationFinalBitwidthConfigVisualizer
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.core.keras.keras_implementation import KerasImplementation
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.core.common.quantization.quantization_config import CustomOpsetLayers
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.attach2keras import \
    AttachTpcToKeras
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import generate_keras_tpc
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import get_op_quantization_configs
from tests.common_tests.helpers.generate_test_tpc import generate_tpc_with_activation_mp
from tests.common_tests.helpers.prep_graph_for_func_test import prepare_graph_set_bit_widths
from model_compression_toolkit.core.common.mixed_precision import MpDistanceWeighting
from model_compression_toolkit.core.common.similarity_analyzer import compute_mse
from tests.keras_tests.tpc_keras import get_tpc_with_activation_mp_keras

keras = tf.keras
layers = keras.layers


def random_datagen():
    return [np.random.random((1, 8, 8, 3))]


def CombinedNMSNet():
    inputs = layers.Input(shape=(8, 8, 3))
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Flatten()(x)

    # Assuming num_boxes is a fixed number, add a reshape here to fit the expected NMS input
    num_boxes = 10  # Example: assuming you are predicting 10 boxes per image
    bbox_output = layers.Dense(num_boxes * 4)(x)  # For bounding box coordinates
    bbox_output = layers.Reshape((num_boxes, 1, 4))(bbox_output)  # Add class dimension

    score_output = layers.Dense(num_boxes)(x)  # For the objectness score of each box
    score_output = layers.Reshape((num_boxes, 1))(score_output)  # Add class dimension

    model = keras.Model(inputs=inputs, outputs=[bbox_output, score_output])

    # Now pass these outputs to NMS
    boxes, scores = model.output
    outputs = tf.image.combined_non_max_suppression(
        boxes=boxes,
        scores=scores,
        max_output_size_per_class=5,
        max_total_size=5,
        iou_threshold=0.5,
        score_threshold=0.5,
        pad_per_class=False,
        clip_boxes=False
    )

    final_model = keras.Model(inputs=inputs, outputs=outputs, name='test_nms')
    return final_model




def SingleOutputNet():
    inputs = layers.Input(shape=(8, 8, 3))
    x = layers.Dense(2)(inputs)
    x = layers.Conv2D(2, 4)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6.0)(x)
    output = layers.Dense(2)(x)
    return keras.Model(inputs=inputs, outputs=output)


def MultipleOutputsNet():
    inputs = layers.Input(shape=(8, 8, 3))
    x = layers.Dense(2)(inputs)
    x = layers.Conv2D(2, 4)(x)
    x = layers.BatchNormalization()(x)
    out1 = layers.ReLU(max_value=6.0)(x)
    out2 = layers.Dense(2)(out1)
    return keras.Model(inputs=inputs, outputs=[out1, out2])


class TestFileLogger(unittest.TestCase):
    """
    This is the test of Keras Logger.
    This test checks logging into file.
    """

    def setUp(self):
        Logger.set_log_file('/tmp/')

    def tensorboard_log_dir(self):
        self.assertTrue(os.path.exists(os.path.join(Logger.LOG_PATH, 'tensorboard_logs')))

    def tensorboard_initial_graph_num_of_nodes(self, num_event_files, event_to_test):
        events_dir = os.path.join(Logger.LOG_PATH, 'tensorboard_logs/')

        initial_graph_events_files = glob.glob(events_dir + 'initial_graph/*events*')
        self.assertTrue(len(initial_graph_events_files) == num_event_files)  # Make sure there is only 2 event files in
        # 'initial_graph' subdir

        initial_graph_event = initial_graph_events_files[event_to_test]

        efl = event_file_loader.LegacyEventFileLoader(initial_graph_event).Load()
        for e in efl:
            if len(e.graph_def) > 0:  # skip events with no graph_def such as event version
                g = GraphDef().FromString(e.graph_def)
        nodes_in_model = len(self.model.layers)
        nodes_in_graph = len(g.node)
        self.assertTrue(nodes_in_graph == nodes_in_model)

    def plot_tensor_sizes(self, core_config):
        model = SingleOutputNet()
        base_config, _, default_config = get_op_quantization_configs()
        tpc_model = generate_tpc_with_activation_mp(
            base_cfg=base_config,
            default_config=default_config,
            mp_bitwidth_candidates_list=[(8, 8), (8, 4), (8, 2),
                                         (4, 8), (4, 4), (4, 2),
                                         (2, 8), (2, 4), (2, 2)])
        tpc = generate_keras_tpc(name='mp_keras_tpc', tpc=tpc_model)
        fqc =AttachTpcToKeras().attach(tpc, core_config.quantization_config.custom_tpc_opset_to_layer)

        # Hessian service assumes core should be initialized. This test does not do it, so we disable the use of hessians in MP
        cfg = mct.core.DEFAULTCONFIG
        mp_cfg = mct.core.MixedPrecisionQuantizationConfig(compute_distance_fn=compute_mse,
                                                           distance_weighting_method=MpDistanceWeighting.AVG,
                                                           use_hessian_based_scores=False)

        # compare max tensor size with plotted max tensor size
        tg = prepare_graph_set_bit_widths(in_model=model,
                                          fw_impl=KerasImplementation(),
                                          fw_info=DEFAULT_KERAS_INFO,
                                          representative_data_gen=random_datagen,
                                          fqc=fqc,
                                          network_editor=[],
                                          quant_config=cfg,
                                          target_resource_utilization=mct.core.ResourceUtilization(weights_memory=73,
                                                                                                   activation_memory=191),
                                          n_iter=1,
                                          analyze_similarity=True,
                                          mp_cfg=mp_cfg)
        tensors_sizes = [4.0 * n.get_total_output_params() / 1000000.0
                         for n in tg.get_sorted_activation_configurable_nodes()]  # in MB
        max_tensor_size = max(tensors_sizes)

        # plot tensor sizes
        activation_conf_nodes_bitwidth = tg.get_final_activation_config()
        visual = ActivationFinalBitwidthConfigVisualizer(activation_conf_nodes_bitwidth)
        fig = visual.plot_tensor_sizes(tg)
        figure_max_tensor_size = max([rect._height for rect in fig.axes[0].get_children()[:len(
            activation_conf_nodes_bitwidth)]])
        self.assertTrue(figure_max_tensor_size == max_tensor_size)

    def get_tpc(self):
        cand_list = [(4, 8), (4, 4), (4, 2),
                     (8, 8), (8, 4), (8, 2),
                     (2, 8), (2, 4), (2, 2)]
        base_config, _, default_config = get_op_quantization_configs()
        return get_tpc_with_activation_mp_keras(base_config=base_config,
                                                default_config=default_config,
                                                mp_bitwidth_candidates_list=cand_list,
                                                name="mp_tensorboard_test")

    def test_steps_by_order(self):
        # Test Single Output Mixed Precision model Logger
        self.model = SingleOutputNet()

        def rep_data():
            yield [np.random.randn(1, 8, 8, 3)]

        mp_qc = mct.core.MixedPrecisionQuantizationConfig(num_of_images=1,
                                                          use_hessian_based_scores=False)
        core_config = mct.core.CoreConfig(mixed_precision_config=mp_qc,
                                          quantization_config=
                                          QuantizationConfig(custom_tpc_opset_to_layer=
                                                             {"Input": CustomOpsetLayers([layers.InputLayer])}),
                                          debug_config=mct.core.DebugConfig(analyze_similarity=True))

        quantized_model, _ = mct.ptq.keras_post_training_quantization(self.model,
                                                                      rep_data,
                                                                      target_resource_utilization=mct.core.ResourceUtilization(
                                                                          weights_memory=73,
                                                                          activation_memory=191),
                                                                      core_config=core_config,
                                                                      target_platform_capabilities=self.get_tpc())

        self.tensorboard_initial_graph_num_of_nodes(num_event_files=1, event_to_test=0)

        # Test Logger file created
        self.tensorboard_log_dir()

        # Test Multiple Outputs model Logger
        self.model = MultipleOutputsNet()
        quantized_model, _ = mct.ptq.keras_post_training_quantization(self.model,
                                                                      rep_data,
                                                                      target_resource_utilization=mct.core.ResourceUtilization(weights_memory=73,
                                                                                                                               activation_memory=191),
                                                                      core_config=core_config,
                                                                      target_platform_capabilities=self.get_tpc())

        # Test tensor size plotting
        self.plot_tensor_sizes(core_config)

        # Disable Logger
        Logger.LOG_PATH = None


class TestLoggerWithNMS(unittest.TestCase):

    def test_logging_with_nms_layer(self):
        Logger.set_log_file('/tmp/')
        self.model = CombinedNMSNet()
        quantized_model, _ = mct.ptq.keras_post_training_quantization(self.model, random_datagen)
        # Disable Logger
        Logger.LOG_PATH = None

if __name__ == '__main__':
    unittest.main()

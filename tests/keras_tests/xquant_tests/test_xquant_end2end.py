#  Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==============================================================================
import sys
import subprocess

import glob
from tensorboard.backend.event_processing import event_file_loader
from tensorboard.compat.proto.graph_pb2 import GraphDef

import os

import unittest
from functools import partial
import tempfile

import model_compression_toolkit as mct
from mct_quantizers import KerasQuantizationWrapper
from model_compression_toolkit.constants import TENSORFLOW
from model_compression_toolkit.target_platform_capabilities.constants import IMX500_TP_MODEL
from model_compression_toolkit.xquant.common.similarity_functions import DEFAULT_SIMILARITY_METRICS_NAMES
from model_compression_toolkit.xquant.common.xquant_config import XQuantConfig

import tensorflow as tf

import keras
import numpy as np

from model_compression_toolkit.xquant.common.constants import OUTPUT_SIMILARITY_METRICS_REPR, \
    OUTPUT_SIMILARITY_METRICS_VAL, INTERMEDIATE_SIMILARITY_METRICS_REPR, \
    INTERMEDIATE_SIMILARITY_METRICS_VAL, XQUANT_REPR, XQUANT_VAL, CUT_MEMORY_ELEMENTS, CUT_TOTAL_SIZE

from model_compression_toolkit.xquant.keras.facade_xquant_report import xquant_report_keras_experimental


def random_data_gen(shape=(8, 8, 3), use_labels=False, num_inputs=1, batch_size=2, num_iter=2):
    if use_labels:
        for _ in range(num_iter):
            yield [[np.random.randn(batch_size, *shape)] * num_inputs, np.random.randn(batch_size)]
    else:
        for _ in range(num_iter):
            yield [np.random.randn(batch_size, *shape)] * num_inputs


class BaseTestEnd2EndKerasXQuant(unittest.TestCase):

    def setUp(self):
        self.float_model = self.get_model_to_test()
        self.repr_dataset = partial(random_data_gen, shape=self.get_input_shape())
        self.quantized_model, _ = mct.ptq.keras_post_training_quantization(in_model=self.float_model,
                                                                           representative_data_gen=self.repr_dataset,
                                                                           core_config=self.get_core_config(),
                                                                           target_platform_capabilities=self.get_tpc())

        self.validation_dataset = partial(random_data_gen, use_labels=True, shape=self.get_input_shape())
        self.tmpdir = tempfile.mkdtemp()
        self.xquant_config = XQuantConfig(report_dir=self.tmpdir)

    def get_input_shape(self):
        return (8, 8, 3)

    def get_core_config(self):
        return mct.core.CoreConfig(debug_config=mct.core.DebugConfig(simulate_scheduler=True))

    def get_tpc(self):
        return mct.get_target_platform_capabilities(TENSORFLOW, IMX500_TP_MODEL, "v2")

    def get_model_to_test(self):
        inputs = keras.layers.Input(shape=self.get_input_shape())
        x = keras.layers.Conv2D(3, 3)(inputs)
        x = keras.layers.BatchNormalization()(x)
        outputs = keras.layers.Activation('relu')(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model

    def test_xquant_report_output_metrics(self):
        self.xquant_config.custom_similarity_metrics = None
        result = xquant_report_keras_experimental(
            self.float_model,
            self.quantized_model,
            self.repr_dataset,
            self.validation_dataset,
            self.xquant_config
        )

        self.assertIn(OUTPUT_SIMILARITY_METRICS_REPR, result)
        self.assertEqual(len(result[OUTPUT_SIMILARITY_METRICS_REPR]), len(DEFAULT_SIMILARITY_METRICS_NAMES))
        self.assertIn(OUTPUT_SIMILARITY_METRICS_VAL, result)
        self.assertEqual(len(result[OUTPUT_SIMILARITY_METRICS_VAL]), len(DEFAULT_SIMILARITY_METRICS_NAMES))

    def test_custom_metric(self):
        self.xquant_config.custom_similarity_metrics = {'mae': lambda x, y: float(tf.keras.losses.MAE(x.flatten(), y.flatten()).numpy())}
        result = xquant_report_keras_experimental(
            self.float_model,
            self.quantized_model,
            self.repr_dataset,
            self.validation_dataset,
            self.xquant_config
        )

        self.assertIn(OUTPUT_SIMILARITY_METRICS_REPR, result)
        self.assertEqual(len(result[OUTPUT_SIMILARITY_METRICS_REPR]), len(DEFAULT_SIMILARITY_METRICS_NAMES) + 1)
        self.assertIn("mae", result[OUTPUT_SIMILARITY_METRICS_REPR])

        self.assertIn(INTERMEDIATE_SIMILARITY_METRICS_REPR, result)
        for k, v in result[INTERMEDIATE_SIMILARITY_METRICS_REPR].items():
            self.assertEqual(len(v), len(DEFAULT_SIMILARITY_METRICS_NAMES) + 1)
            self.assertIn("mae", v)

    def test_intermediate_metrics(self):
        self.xquant_config.custom_similarity_metrics = None
        result = xquant_report_keras_experimental(
            self.float_model,
            self.quantized_model,
            self.repr_dataset,
            self.validation_dataset,
            self.xquant_config
        )

        self.assertIn(INTERMEDIATE_SIMILARITY_METRICS_REPR, result)
        linear_layers = [l.name for l in self.quantized_model.layers if isinstance(l, KerasQuantizationWrapper)]
        self.assertIn(linear_layers[0], result[INTERMEDIATE_SIMILARITY_METRICS_REPR])

        for k, v in result[INTERMEDIATE_SIMILARITY_METRICS_REPR].items():
            self.assertEqual(len(v), len(DEFAULT_SIMILARITY_METRICS_NAMES))

        self.assertIn(INTERMEDIATE_SIMILARITY_METRICS_VAL, result)
        for k, v in result[INTERMEDIATE_SIMILARITY_METRICS_VAL].items():
            self.assertEqual(len(v), len(DEFAULT_SIMILARITY_METRICS_NAMES))

    def test_tensorboard_graph(self):
        self.xquant_config.custom_similarity_metrics = None
        result = xquant_report_keras_experimental(
            self.float_model,
            self.quantized_model,
            self.repr_dataset,
            self.validation_dataset,
            self.xquant_config
        )
        events_dir = os.path.join(self.xquant_config.report_dir, 'xquant')
        initial_graph_events_files = glob.glob(events_dir + '/*events*')
        initial_graph_event = initial_graph_events_files[0]
        efl = event_file_loader.LegacyEventFileLoader(initial_graph_event).Load()
        for e in efl:
            if len(e.graph_def) > 0:  # skip events with no graph_def such as event version
                g = GraphDef().FromString(e.graph_def)
        for node in g.node:
            if node.device == 'KerasQuantizationWrapper':
                self.assertIn(XQUANT_REPR, str(node))
                self.assertIn(XQUANT_VAL, str(node))
                self.assertIn(CUT_MEMORY_ELEMENTS, str(node))
                self.assertIn(CUT_TOTAL_SIZE, str(node))


# Test with Conv2D without BatchNormalization and without Activation
class TestXQuantReportModel2(BaseTestEnd2EndKerasXQuant):
    def get_model_to_test(self):
        inputs = keras.layers.Input(shape=self.get_input_shape())
        x = keras.layers.Conv2D(3, 3, padding='same')(inputs)
        x = keras.layers.Add()([x, inputs])
        outputs = keras.layers.Activation('softmax')(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model


# Test with Multiple Convolution Layers and an Addition Operator
class TestXQuantReportModel3(BaseTestEnd2EndKerasXQuant):
    def get_model_to_test(self):
        inputs = keras.layers.Input(shape=self.get_input_shape())
        x = keras.layers.Conv2D(3, 3)(inputs)
        y = keras.layers.Conv2D(3, 3)(inputs)
        x = keras.layers.Add()([x, y])  # Adding operator
        x = keras.layers.Conv2D(3, 3)(x)
        outputs = keras.layers.Activation('softmax')(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model


class TestXQuantReportModelMBv2(BaseTestEnd2EndKerasXQuant):
    def get_input_shape(self):
        return (224, 224, 3)

    def get_model_to_test(self):
        from keras.applications.mobilenet_v2 import MobileNetV2
        model = MobileNetV2()
        return model


class TestXQuantReportModelMBv1(BaseTestEnd2EndKerasXQuant):
    def get_input_shape(self):
        return (224, 224, 3)

    def get_model_to_test(self):
        from keras.applications.mobilenet import MobileNet
        model = MobileNet()
        return model

if __name__ == '__main__':
    unittest.main()

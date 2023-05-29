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
import os
import tempfile
import unittest

import keras.models
import numpy as np
import tensorflow as tf
from keras import Input
from keras.layers import Conv2D, BatchNormalization, ReLU, Dropout, Dense, Activation

import model_compression_toolkit as mct
from model_compression_toolkit.exporter.model_wrapper import is_keras_layer_exportable
from model_compression_toolkit.trainable_infrastructure.keras.load_model import \
    keras_load_quantized_model

from model_compression_toolkit.target_platform_capabilities.constants import DEFAULT_TP_MODEL
from model_compression_toolkit.constants import TENSORFLOW
from model_compression_toolkit import get_target_platform_capabilities
from tests.keras_tests.utils import get_layers_from_model_by_type

DEFAULT_KERAS_TPC = get_target_platform_capabilities(TENSORFLOW, DEFAULT_TP_MODEL)

_, SAVED_EXPORTABLE_MODEL_PATH_TF = tempfile.mkstemp('.h5')
_, SAVED_MODEL_PATH_TF = tempfile.mkstemp('.h5')
_, SAVED_MODEL_PATH_TFLITE = tempfile.mkstemp('.tflite')


def _get_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv2D(7, 7)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(3, 3)(x)
    x = Activation('swish')(x)
    x = Conv2D(3, 3, padding='same')(x)
    x = ReLU(max_value=6)(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(10)(x)
    return keras.Model(inputs=inputs, outputs=x)


class TestKerasFakeQuantExporter(unittest.TestCase):

    def tearDown(self):
        os.remove(SAVED_MODEL_PATH_TF)
        os.remove(SAVED_MODEL_PATH_TFLITE)
        os.remove(SAVED_EXPORTABLE_MODEL_PATH_TF)

    def test_export_tf_phases(self) -> None:
        input_shape = (32, 32, 3)
        self.model = _get_model(input_shape)
        self.representative_data_gen = lambda: [np.random.randn(*((1,) + input_shape))]
        self.exportable_model = self.run_mct(self.model, new_experimental_exporter=True)
        self.exportable_model.trainable = False
        self.save_and_load_exportable_model()
        self.save_and_load_exported_tf_fakequant_model()
        self.save_and_load_exported_tflite_fakequant_model()
        self.equal_predictions_between_exportable_and_tf_fq_exported()
        self.tflite_fq_exported_weights()

    def run_mct(self, model, new_experimental_exporter):
        core_config = mct.core.CoreConfig()
        self.tpc = DEFAULT_KERAS_TPC
        new_export_model, _ = mct.ptq.keras_post_training_quantization_experimental(
            in_model=model,
            core_config=core_config,
            representative_data_gen=self.representative_data_gen,
            target_platform_capabilities=self.tpc,
            new_experimental_exporter=new_experimental_exporter)
        return new_export_model

    def save_and_load_exportable_model(self):
        self.exportable_model.save(SAVED_EXPORTABLE_MODEL_PATH_TF)
        keras_load_quantized_model(SAVED_EXPORTABLE_MODEL_PATH_TF)

    def save_and_load_exported_tf_fakequant_model(self):
        mct.exporter.keras_export_model(model=self.exportable_model,
                                        is_layer_exportable_fn=is_keras_layer_exportable,
                                        save_model_path=SAVED_MODEL_PATH_TF,
                                        target_platform_capabilities=self.tpc,
                                        serialization_format=mct.exporter.KerasExportSerializationFormat.KERAS_H5)
        keras_load_quantized_model(SAVED_MODEL_PATH_TF)

    def save_and_load_exported_tflite_fakequant_model(self):
        mct.exporter.keras_export_model(model=self.exportable_model,
                                        is_layer_exportable_fn=is_keras_layer_exportable,
                                        save_model_path=SAVED_MODEL_PATH_TFLITE,
                                        target_platform_capabilities=self.tpc,
                                        serialization_format=mct.exporter.KerasExportSerializationFormat.TFLITE)
        interpreter = tf.lite.Interpreter(model_path=SAVED_MODEL_PATH_TFLITE)
        interpreter.allocate_tensors()

    def equal_predictions_between_exportable_and_tf_fq_exported(self):
        """
        Test that the exported model and fully quantized model predicting the same results.
        """
        images = self.representative_data_gen()
        diff = keras_load_quantized_model(SAVED_MODEL_PATH_TF)(images) - self.exportable_model(images)
        self.assertTrue(np.sum(np.abs(diff)) == 0)

    def tflite_fq_exported_weights(self):
        """
        Test that the tflite exported model can infer and that tf exported model has the same weights
        as the tflite exported model.
        """
        tf_exported_model = keras_load_quantized_model(SAVED_MODEL_PATH_TF)

        # Test inference of exported model
        test_image = self.representative_data_gen()[0].astype("float32")
        interpreter = tf.lite.Interpreter(model_path=SAVED_MODEL_PATH_TFLITE)
        interpreter.allocate_tensors()
        input_index = interpreter.get_input_details()[0]["index"]
        interpreter.set_tensor(input_index, test_image)
        # Run inference.
        interpreter.invoke()

        # Since the tensor index and name in tflite may change between tf versions,
        # we're looking for a tensor with a specific shape in tensor details
        tflite_kernel_shape = [7, 7, 7, 3]

        kernel_tensor_index, bias_tensor_index = None, None
        for d in interpreter.get_tensor_details():
            if np.all(d['shape'] == tflite_kernel_shape):
                kernel_tensor_index = d['index']
                break

        assert kernel_tensor_index is not None, f'did not find the kernel tensor index'

        # Test equal weights of the first conv2d layer between exported TFLite and exported TF
        conv_layer = get_layers_from_model_by_type(tf_exported_model, Conv2D)[0]
        diff = np.transpose(interpreter.tensor(kernel_tensor_index)(), (1, 2, 3, 0)) - conv_layer.kernel
        self.assertTrue(np.sum(np.abs(diff)) == 0)

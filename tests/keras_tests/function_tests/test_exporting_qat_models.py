# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
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

import tempfile
import unittest

import keras
import numpy as np
import tensorflow as tf
from keras.layers import TFOpLambda

import model_compression_toolkit as mct
from mct_quantizers import KerasActivationQuantizationHolder
from tests.keras_tests.exporter_tests.keras_fake_quant.keras_fake_quant_exporter_base_test import \
    get_minmax_from_qparams
from tests.keras_tests.utils import get_layers_from_model_by_type
from mct_quantizers import keras_load_quantized_model as mct_quantizers_load # Load with custom of inferable objects
from model_compression_toolkit import keras_load_quantized_model as mct_load # Load with custom of inferable+trainable objects

layers = keras.layers




class TestExportingQATModelBase(unittest.TestCase):

    def get_model(self):
        i = keras.Input(shape=(224,224,3))
        x = layers.Conv2D(3,3)(i)
        model = keras.models.Model(i,x)
        return model

    def get_dataset(self):
        yield [np.random.rand(1, 224, 224, 3)]

    def get_tpc(self):
        return mct.get_target_platform_capabilities('tensorflow','default')

    def get_serialization_format(self):
        return mct.exporter.KerasExportSerializationFormat.KERAS_H5

    def get_filepath(self):
        return tempfile.mkstemp('.h5')[1]

    def load_exported_model(self, filepath):
        return mct_quantizers_load(filepath)

    def infer(self, model, images):
        return model(images)

    def export_qat_model(self):
        model = self.get_model()
        images = next(self.get_dataset())

        self.qat_ready, _, _ = mct.qat.keras_quantization_aware_training_init(model,
                                                                         self.get_dataset)
        _qat_ready_model_path = tempfile.mkstemp('.h5')[1]
        keras.models.save_model(self.qat_ready, _qat_ready_model_path)
        self.qat_ready = mct_load(_qat_ready_model_path)

        qat_ready_pred = self.qat_ready(images)
        self.final_model = mct.qat.keras_quantization_aware_training_finalize(self.qat_ready)

        _finalized_model_path = tempfile.mkstemp('.h5')[1]
        keras.models.save_model(self.final_model, _finalized_model_path)
        self.final_model = mct_quantizers_load(_finalized_model_path)

        qat_final_pred = self.final_model(images)
        diff = np.sum(np.abs(qat_ready_pred - qat_final_pred))
        assert diff == 0, f'QAT Model before and after finalizing should predict' \
                          f' identical predictions but diff is ' \
                          f'{diff}'

        self.filepath = self.get_filepath()
        mct.exporter.keras_export_model(self.final_model,
                                        self.filepath,
                                        self.get_tpc(),
                                        serialization_format=self.get_serialization_format())

        self.loaded_model = self.load_exported_model(self.filepath)
        self.infer(self.loaded_model, images)

    def test_exported_qat_model(self):
        self.export_qat_model()
        images = next(self.get_dataset())
        a = self.infer(self.final_model, images)
        b = self.infer(self.loaded_model, images)
        diff = np.max(np.abs(a-b))
        assert diff == 0, f'QAT Model before and after export to h5 should ' \
                          f'predict identical predictions but diff is ' \
                          f'{diff}'

        holder_layers_finalized_model = get_layers_from_model_by_type(self.final_model, KerasActivationQuantizationHolder)
        holder_layers_loaded_model = get_layers_from_model_by_type(self.loaded_model, TFOpLambda)

        _finalized_first_activation_q_params = holder_layers_finalized_model[0].get_config()['activation_holder_quantizer']['config']
        _min, _max = get_minmax_from_qparams(_finalized_first_activation_q_params)
        self.assertTrue(_min == holder_layers_loaded_model[0].inbound_nodes[0].call_kwargs['min'])
        self.assertTrue(_max == holder_layers_loaded_model[0].inbound_nodes[0].call_kwargs['max'])
        self.assertTrue(_finalized_first_activation_q_params['num_bits'] == holder_layers_loaded_model[0].inbound_nodes[0].call_kwargs['num_bits'])

        _finalized_second_activation_q_params = holder_layers_finalized_model[1].get_config()['activation_holder_quantizer']['config']
        _min, _max = get_minmax_from_qparams(_finalized_second_activation_q_params)
        self.assertTrue(_min == holder_layers_loaded_model[1].inbound_nodes[0].call_kwargs['min'])
        self.assertTrue(_max == holder_layers_loaded_model[1].inbound_nodes[0].call_kwargs['max'])
        self.assertTrue(_finalized_second_activation_q_params['num_bits'] == holder_layers_loaded_model[1].inbound_nodes[0].call_kwargs['num_bits'])

        conv_finalized_model = get_layers_from_model_by_type(self.final_model, layers.Conv2D)[0]
        conv_loaded_model = get_layers_from_model_by_type(self.loaded_model, layers.Conv2D)[0]
        self.assertTrue(np.all(conv_finalized_model.get_quantized_weights()['kernel']==conv_loaded_model.kernel))


class TestExportingQATModelTFLite(TestExportingQATModelBase):

    def get_serialization_format(self):
        return mct.exporter.KerasExportSerializationFormat.TFLITE

    def get_filepath(self):
        return tempfile.mkstemp('.tflite')[1]

    def load_exported_model(self, filepath):
        # Load model
        interpreter = tf.lite.Interpreter(model_path=filepath)
        interpreter.allocate_tensors()
        return interpreter

    def infer(self, model, images):
        input_index = model.get_input_details()[0]['index']
        model.set_tensor(input_index, images[0].astype("float32"))
        model.invoke()
        output_details = model.get_output_details()
        output_data = model.get_tensor(output_details[0]['index'])
        return output_data

    def test_exported_qat_model(self):
        self.export_qat_model()
        # Compare scales from tflite model to the fully-quantized model
        exported_model_scales=[]
        for op in self.loaded_model._get_ops_details():
            if op['op_name'] == 'QUANTIZE':
                # Take scale from quant params of the output tensor of QUANTIZE op
                exported_model_scales.append(self.loaded_model._get_tensor_details(op['outputs'][0])['quantization_parameters']['scales'][0])

        # Get Kernel values
        tflite_kernel=None
        for t in self.loaded_model.get_tensor_details():
            if np.all(t['shape'] == np.asarray([3, 3, 3, 3])) and len(t['shape']) == 4:
                tflite_kernel = self.loaded_model.tensor(t['index'])()
        assert tflite_kernel is not None, f' Could not find conv kernel in tflite model'

        def _get_activation_scale(fq_args):
            if fq_args['signed']:
                return fq_args['threshold'][0] / (2 ** (fq_args['num_bits'] - 1))
            return fq_args['threshold'][0] / (2 ** fq_args['num_bits'])

        holder_layers = get_layers_from_model_by_type(self.final_model, KerasActivationQuantizationHolder)
        first_fq_args = holder_layers[0].activation_holder_quantizer.get_config()
        first_scale = _get_activation_scale(first_fq_args)
        second_fq_args = holder_layers[1].activation_holder_quantizer.get_config()
        second_scale = _get_activation_scale(second_fq_args)
        self.assertTrue(first_scale == exported_model_scales[0],
                        f'Expect same scales from exported tflite model and QAT finalized model but found '
                        f'{first_scale} and {exported_model_scales[0]}')
        self.assertTrue(second_scale == exported_model_scales[1],
                        f'Expect same scales from exported tflite model and QAT finalized model but found '
                        f'{second_scale} and {exported_model_scales[1]}')

        conv_layer = get_layers_from_model_by_type(self.final_model, layers.Conv2D)[0]
        finalized_kernel = conv_layer.get_quantized_weights()['kernel']
        kernels_diff=np.max(np.abs(finalized_kernel-tflite_kernel.transpose((1,2,3,0))))
        self.assertTrue(kernels_diff==0, f'Kernels should be identical but diff of {kernels_diff} was found')






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

import onnx
import onnxruntime
from onnx import numpy_helper

import model_compression_toolkit as mct
from tests.pytorch_tests.exporter_tests.base_pytorch_export_test import BasePytorchExportTest


class BasePytorchONNXExportTest(BasePytorchExportTest):

    def get_serialization_format(self):
        return mct.exporter.PytorchExportSerializationFormat.ONNX

    def get_tmp_filepath(self):
        return tempfile.mkstemp('.onnx')[1]

    def load_exported_model(self, filepath):
        # Check that the model is well formed
        onnx.checker.check_model(filepath)
        return onnx.load(filepath)

    def infer(self, model, images):
        ort_session = onnxruntime.InferenceSession(model.SerializeToString())

        def to_numpy(tensor):
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

        # compute ONNX Runtime output prediction
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(images)}
        onnx_output = ort_session.run(None, ort_inputs)
        return onnx_output[0]

    def _get_onnx_node_by_type(self, onnx_model, op_type):
        return [n for n in onnx_model.graph.node if n.op_type == op_type]

    def _get_onnx_node_const_inputs(self, onnx_model, op_type):
        constname_to_constvalue = {node.output[0]: numpy_helper.to_array(node.attribute[0].t) for node in
                                   onnx_model.graph.node if node.op_type == 'Constant'}
        q_nodes = [n for n in onnx_model.graph.node if n.op_type == op_type]
        assert len(q_nodes) == 1
        node = q_nodes[0]
        node_qparams = [constname_to_constvalue[input_name] for input_name in node.input if
                        input_name in constname_to_constvalue]

        return node_qparams

    def _get_onnx_node_attributes(self, onnx_node):
        # Extract attributes as a key-value dictionary
        attributes_dict = {}
        for attribute in onnx_node.attribute:
            if attribute.HasField('f'):
                attributes_dict[attribute.name] = attribute.f
            elif attribute.HasField('i'):
                attributes_dict[attribute.name] = attribute.i
            elif attribute.HasField('s'):
                attributes_dict[attribute.name] = attribute.s.decode('utf-8')
            else:
                raise Exception(f'Encountered an unfamiliar attribute type in attribute {attribute}')

        return attributes_dict

class BasePytorchONNXCustomOpsExportTest(BasePytorchONNXExportTest):

    def run_export(self, quantized_model, use_onnx_custom_ops=True):
        super().run_export(quantized_model,
                           use_onnx_custom_ops=use_onnx_custom_ops)

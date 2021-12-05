# Copyright 2021 Sony Semiconductors Israel, Inc. All rights reserved.
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
import unittest

from model_compression_toolkit.keras.back2framework.model_builder import model_builder
from model_compression_toolkit.common.model_builder_mode import ModelBuilderMode
from model_compression_toolkit.common.substitutions.apply_substitutions import substitute
from model_compression_toolkit.keras.keras_implementation import KerasImplementation
from model_compression_toolkit.keras.reader.reader import model_reader
from tests.common_tests.helpers.tensors_compare import cosine_similarity

keras = tf.keras
layers = keras.layers


class NetworkTest(object):
    def __init__(self, unit_test, model_float, input_shapes):
        self.unit_test = unit_test
        self.model_float = model_float
        self.input_shapes = input_shapes

    def compare(self, inputs_list, quantized_model):
        output_q = quantized_model.predict(inputs_list)
        output_f = self.model_float.predict(inputs_list)
        if isinstance(output_f, list):
            cs = np.mean([cosine_similarity(oq, of) for oq, of, in zip(output_q, output_f)])
        else:
            cs = cosine_similarity(output_f, output_q)

        self.unit_test.assertTrue(np.isclose(cs, 1, 0.001), msg=f'fail cosine similarity check:{cs}')

    def run_network(self, inputs_list):
        fw_impl = KerasImplementation()
        graph = model_reader(self.model_float)  # model reading
        ptq_model, _ = model_builder(graph,
                                     mode=ModelBuilderMode.FLOAT)
        self.compare(inputs_list, ptq_model)

        ptq_model, _ = model_builder(substitute(graph,
                                                fw_impl.get_substitutions_pre_statistics_collection()),
                                     mode=ModelBuilderMode.FLOAT)
        self.compare(inputs_list, ptq_model)

    @staticmethod
    def create_inputs(inputs_list):
        return [np.random.randn(*in_shape) for in_shape in inputs_list]


class FeatureNetworkFloatTest(unittest.TestCase):

    def run_network(self, model_float, input_shapes):
        inputs_list = NetworkTest.create_inputs(input_shapes)
        NetworkTest(self, model_float, input_shapes).run_network(inputs_list)

    def test_mobilenet_v1(self):
        input_shapes = [[32, 224, 224, 3]]
        from tensorflow.keras.applications.mobilenet import MobileNet
        self.run_network(MobileNet(), input_shapes)

    def test_mobilenet_v1_gptq(self):
        input_shapes = [[16, 224, 224, 3]]
        from tensorflow.keras.applications.mobilenet import MobileNet
        self.run_network(MobileNet(), input_shapes)

    def test_mobilenet_v2(self):
        input_shapes = [[1, 224, 224, 3]]
        from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
        self.run_network(MobileNetV2(), input_shapes)

    def test_xception(self):
        input_shapes = [[1, 299, 299, 3]]
        from tensorflow.keras.applications.xception import Xception
        self.run_network(Xception(), input_shapes)

    def test_resnet(self):
        input_shapes = [[1, 224, 224, 3]]
        from tensorflow.keras.applications.resnet import ResNet50
        self.run_network(ResNet50(), input_shapes)

    def test_efficientnetbo(self):
        input_shapes = [[4, 224, 224, 3]]
        from tensorflow.keras.applications.efficientnet import EfficientNetB0
        self.run_network(EfficientNetB0(), input_shapes)

    def test_nasnetmobile(self):
        input_shapes = [[1, 224, 224, 3]]
        from tensorflow.keras.applications.nasnet import NASNetMobile
        self.run_network(NASNetMobile(), input_shapes)

    def test_nasnetlarge(self):
        input_shapes = [[4, 331, 331, 3]]
        from tensorflow.keras.applications.nasnet import NASNetLarge
        self.run_network(NASNetLarge(), input_shapes)

    def test_resnetv2(self):
        input_shapes = [[1, 224, 224, 3]]
        from tensorflow.keras.applications.resnet_v2 import ResNet50V2
        self.run_network(ResNet50V2(), input_shapes)

    def test_densenet121(self):
        input_shapes = [[1, 224, 224, 3]]
        from tensorflow.keras.applications.densenet import DenseNet121
        self.run_network(DenseNet121(), input_shapes)

    def test_vgg(self):
        input_shapes = [[1, 224, 224, 3]]
        from tensorflow.keras.applications.vgg16 import VGG16
        self.run_network(VGG16(), input_shapes)

    def test_inceptionresnet(self):
        input_shapes = [[1, 299, 299, 3]]
        from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
        self.run_network(InceptionResNetV2(), input_shapes)

    def test_inception(self):
        input_shapes = [[1, 299, 299, 3]]
        from tensorflow.keras.applications.inception_v3 import InceptionV3
        self.run_network(InceptionV3(), input_shapes)


if __name__ == '__main__':
    unittest.main()

import unittest

import keras

from tests.keras_tests.exporter_tests.tflite_int8.networks.conv2d_test import TestConv2DExporter
from tests.keras_tests.exporter_tests.tflite_int8.networks.dense_test import TestDenseExporter
from tests.keras_tests.exporter_tests.tflite_int8.networks.depthwiseconv2d_test import TestDepthwiseConv2DExporter
from tests.keras_tests.exporter_tests.tflite_int8.networks.mobilenetv2_test import TestMBV2Exporter



class TFLiteINT8ExporterNetworksTest(unittest.TestCase):

    def test_conv2d(self):
        TestConv2DExporter().run_test()

    def test_depthwiseconv2d(self):
        TestDepthwiseConv2DExporter().run_test()

    def test_dense(self):
        TestDenseExporter().run_test()

    def test_mbv2(self):
        TestMBV2Exporter().run_test()





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
import copy
import unittest
from functools import partial
import doctest

import numpy as np
import tensorflow as tf

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
if tf.__version__ < "2.6":
    from tensorflow.keras.layers import Conv2D, Conv2DTranspose, ReLU, Activation, Input
else:
    from keras.layers import Conv2D, Conv2DTranspose, ReLU, Activation
    from keras import Input

import model_compression_toolkit as mct
from model_compression_toolkit import ptq
from model_compression_toolkit import gptq
from model_compression_toolkit import core


class TestKerasDocsExamples(unittest.TestCase):

    def test_keras_ptq_facade(self):
        doctest.testfile("quantization_facade.py", package=ptq.keras, verbose=True, raise_on_error=True)

    def test_keras_gptq_facade(self):
        doctest.testfile("quantization_facade.py", package=gptq.keras, verbose=True, raise_on_error=True)

    def test_keras_kpi_data_facade(self):
        doctest.testfile("kpi_data_facade.py", package=core.keras, verbose=True, raise_on_error=True)

    def test_pytorch_ptq_facade(self):
        doctest.testfile("quantization_facade.py", package=ptq.pytorch, verbose=True, raise_on_error=True)

    def test_pytorch_gptq_facade(self):
        doctest.testfile("quantization_facade.py", package=gptq.pytorch, verbose=True, raise_on_error=True)

    def test_pytorch_kpi_data_facade(self):
        doctest.testfile("kpi_data_facade.py", package=core.pytorch, verbose=True, raise_on_error=True)


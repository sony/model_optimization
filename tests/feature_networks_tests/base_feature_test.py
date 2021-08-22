# ===============================================================================
# Copyright (c) 2021, Sony Semiconductors Israel, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ===============================================================================


import numpy as np

import network_optimization_package as snop
from network_optimization_package.common.quantization.quantization_config import DEFAULTCONFIG
from network_optimization_package.keras.default_framework_info import DEFAULT_KERAS_INFO


class BaseFeatureNetworkTest:
    def __init__(self, unit_test, num_calibration_iter=1, val_batch_size=50):
        self.unit_test = unit_test
        self.val_batch_size = val_batch_size
        self.num_calibration_iter = num_calibration_iter

    def get_quantization_config(self):
        return DEFAULTCONFIG

    def get_kd_config(self):
        return None

    def get_network_editor(self):
        return []

    def create_inputs_shape(self):
        raise NotImplementedError(f'{self.__class__} did not implement create_feature_network')

    def create_feature_network(self, input_shape):
        raise NotImplementedError(f'{self.__class__} did not implement create_feature_network')

    def compare(self, ptq_model, model_float, input_x=None):
        raise NotImplementedError(f'{self.__class__} did not implement compare')

    @staticmethod
    def generate_inputs(input_shapes):
        return [np.random.randn(*in_shape) for in_shape in input_shapes]

    def run_test(self):
        input_shapes = self.create_inputs_shape()
        x = self.generate_inputs(input_shapes)

        def representative_data_gen():
            return x

        model_float = self.create_feature_network(input_shapes)
        ptq_model, quantization_info = snop.keras_post_training_quantization(model_float, representative_data_gen,
                                                                             n_iter=self.num_calibration_iter,
                                                                             quant_config=self.get_quantization_config(),
                                                                             fw_info=DEFAULT_KERAS_INFO,
                                                                             network_editor=self.get_network_editor(),
                                                                             knowledge_distillation_config=self.get_kd_config())
        self.compare(ptq_model, model_float, input_x=x, quantization_info=quantization_info)

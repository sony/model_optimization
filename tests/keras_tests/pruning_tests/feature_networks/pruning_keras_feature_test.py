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
import model_compression_toolkit as mct
from model_compression_toolkit.core.common.pruning.pruning_config import PruningConfig
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
import numpy as np

class PruningKerasFeatureTest(BaseKerasFeatureNetworkTest):
    def __init__(self,
                 unit_test,
                 num_calibration_iter=1,
                 val_batch_size=1,
                 num_of_inputs=1,
                 input_shape=(8, 8, 3)):

        super().__init__(unit_test=unit_test,
                         val_batch_size=val_batch_size,
                         num_calibration_iter=num_calibration_iter,
                         num_of_inputs=num_of_inputs,
                         input_shape=input_shape)

        self.dense_model_num_params = None

    def get_pruning_config(self):
        return PruningConfig(num_score_approximations=1)

    def run_test(self):
        feature_networks = self.create_networks()
        feature_networks = feature_networks if isinstance(feature_networks, list) else [feature_networks]
        for model_float in feature_networks:
            self.dense_model_num_params=sum([l.count_params() for l in model_float.layers])
            pruned_model, pruning_info = mct.pruning.keras_pruning_experimental(model=model_float,
                                                                                target_kpi=self.get_kpi(),
                                                                                representative_data_gen=self.representative_data_gen_experimental,
                                                                                pruning_config=self.get_pruning_config(),
                                                                                target_platform_capabilities=self.get_tpc())

            self.pruned_model_num_params=sum([l.count_params() for l in pruned_model.layers])

            ### Test inference ##
            input_tensor = self.representative_data_gen()
            pruned_outputs = pruned_model(input_tensor)
            if self.pruned_model_num_params == self.dense_model_num_params:
                dense_outputs = model_float(input_tensor)
                assert np.sum(np.abs(dense_outputs-pruned_outputs))==0

            assert pruned_model.output_shape==model_float.output_shape
            for dense_layer, pruned_layer in zip(model_float.layers, pruned_model.layers):
                assert type(pruned_layer)==type(dense_layer)

            self.compare(pruned_model,
                         model_float,
                         input_x=input_tensor,
                         quantization_info=pruning_info)


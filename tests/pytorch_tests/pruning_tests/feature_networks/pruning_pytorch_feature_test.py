# Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
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
import torch.nn

import model_compression_toolkit as mct
from model_compression_toolkit.core.common.pruning.pruning_config import PruningConfig
import numpy as np

from model_compression_toolkit.core.pytorch.pytorch_device_config import get_working_device
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor, torch_tensor_to_numpy
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import generate_pytorch_tpc
from tests.common_tests.helpers.generate_test_tp_model import generate_test_tp_model
from tests.pytorch_tests.model_tests.base_pytorch_feature_test import BasePytorchFeatureNetworkTest
from tests.pytorch_tests.utils import count_model_prunable_params


class PruningPytorchFeatureTest(BasePytorchFeatureNetworkTest):
    def __init__(self,
                 unit_test,
                 num_calibration_iter=1,
                 val_batch_size=1,
                 num_of_inputs=1,
                 input_shape=(3, 8, 8)):

        super().__init__(unit_test=unit_test,
                         val_batch_size=val_batch_size,
                         num_calibration_iter=num_calibration_iter,
                         num_of_inputs=num_of_inputs,
                         input_shape=input_shape)

        self.use_bn = False

    def get_pruning_config(self):
        return PruningConfig(num_score_approximations=2)

    def get_tpc(self):
        tp = generate_test_tp_model({'simd_size': self.simd})
        return generate_pytorch_tpc(name="simd_test", tp_model=tp)

    def get_kpi(self, dense_model_num_params, model):
        if not self.use_bn and torch.nn.BatchNorm2d in [type(m) for m in model.modules()]:
            # substract the 4 bn params if the bn is not used. This is because Back2Framework will create a model without bn
            dense_model_num_params -= count_model_prunable_params(model.bn)
        # Remove only one group of channels only one parameter should be pruned
        return mct.KPI(weights_memory=(dense_model_num_params-self.simd) * 4)

    def run_test(self):
        feature_networks = self.create_networks()
        feature_networks = feature_networks if isinstance(feature_networks, list) else [feature_networks]
        for model_float in feature_networks:
            model_float.to(get_working_device())
            # self.dense_model_num_params = sum(p.numel() for p in model_float.parameters())
            dense_model_num_params = count_model_prunable_params(model_float)
            pruned_model, pruning_info = mct.pruning.pytorch_pruning_experimental(model=model_float,
                                                                                  target_kpi=self.get_kpi(dense_model_num_params, model_float),
                                                                                  representative_data_gen=self.representative_data_gen_experimental,
                                                                                  pruning_config=self.get_pruning_config(),
                                                                                  target_platform_capabilities=self.get_tpc())

            quantized_model, _ = mct.ptq.pytorch_post_training_quantization_experimental(in_module=model_float,
                                                                                         core_config=mct.core.CoreConfig(),
                                                                                         representative_data_gen=self.representative_data_gen_experimental,
                                                                                         target_platform_capabilities=self.get_tpc())

            pruned_quantized_model, _ = mct.ptq.pytorch_post_training_quantization_experimental(in_module=pruned_model,
                                                                                         core_config=mct.core.CoreConfig(),
                                                                                         representative_data_gen=self.representative_data_gen_experimental,
                                                                                         target_platform_capabilities=self.get_tpc())

            pruned_model_num_params = sum(p.numel() for p in pruned_model.parameters())

            ### Test inference ##
            input_tensor = to_torch_tensor(self.representative_data_gen())
            float_outputs = torch_tensor_to_numpy(model_float(input_tensor[0]))
            pruned_outputs = torch_tensor_to_numpy(pruned_model(input_tensor))
            if pruned_model_num_params == dense_model_num_params:
                dense_outputs = model_float(input_tensor[0])
                self.unit_test.assertTrue(np.sum(np.abs(dense_outputs-pruned_outputs)) == 0, f"If model is not pruned, "
                                                                                           f"predictions should be identical, but found difference between predictions")

            self.unit_test.assertTrue(pruned_outputs.shape == float_outputs.shape,
                                      f"The pruned model's output should have the same output shape as dense model,"
                                      f"but dense model output shape is {float_outputs.shape},"
                                      f"and pruned model output shape is {pruned_outputs.shape}")

            for dense_layer, pruned_layer in zip(quantized_model.named_children(), pruned_quantized_model.named_children()):
                dense_layer_type = type(dense_layer[1])
                pruned_layer_type = type(pruned_layer[1])
                self.unit_test.assertTrue(pruned_layer_type==dense_layer_type, f"type of layers and their orders should be the same,"
                                                                                 f"but {dense_layer_type} is not {pruned_layer_type}")

            self.compare(pruned_model,
                         model_float,
                         input_x=input_tensor,
                         quantization_info=pruning_info)


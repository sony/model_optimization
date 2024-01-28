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

import tempfile
from enum import Enum
import unittest

import model_compression_toolkit as mct
import numpy as np
from packaging import version

from model_compression_toolkit.constants import FP32_BYTES_PER_PARAMETER
from model_compression_toolkit.core.common.pruning.importance_metrics.importance_metric_factory import IMPORTANCE_METRIC_DICT
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor
from tests.keras_tests.pruning_tests.random_importance_metric import RandomImportanceMetric
import torchvision.models as models

from tests.pytorch_tests.utils import count_model_prunable_params

NUM_PRUNING_RATIOS = 1

class TestImportanceMetric(Enum):
    RANDOM = 'random'

IMPORTANCE_METRIC_DICT.update({TestImportanceMetric.RANDOM: RandomImportanceMetric})

class PruningPretrainedModelsTest(unittest.TestCase):
    def representative_dataset(self, in_shape=(1,3, 224,224)):
        for _ in range(1):
            yield [np.random.randn(*in_shape)]

    def test_rn50_pruning(self):
        # Load a pre-trained ResNet50 model
        dense_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        target_crs = np.linspace(0.5, 1, NUM_PRUNING_RATIOS)
        for cr in target_crs:
            self.run_test(cr, dense_model)

    def test_efficientnetb0_pruning(self):
        # Load a pre-trained EfficientNetB0 model
        dense_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        target_crs = np.linspace(0.8, 1, NUM_PRUNING_RATIOS)
        for cr in target_crs:
            self.run_test(cr, dense_model)


    def test_vgg16_pruning(self):
        # Load a pre-trained VGG16 model
        dense_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        target_crs = np.linspace(0.5, 1, NUM_PRUNING_RATIOS)
        for cr in target_crs:
            self.run_test(cr, dense_model)

    def test_mobilenetv2_pruning(self):
        # Load a pre-trained MobileNetV2 model
        dense_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        target_crs = np.linspace(0.5, 1, NUM_PRUNING_RATIOS)
        for cr in target_crs:
            self.run_test(cr, dense_model)

    def test_densenet_pruning(self):
        # Load a pre-trained DenseNet121 model
        dense_model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        target_crs = np.linspace(0.5, 1, NUM_PRUNING_RATIOS)
        for cr in target_crs:
            self.run_test(cr, dense_model)


    def test_vgg19_pruning(self):
        # Load a pre-trained VGG19 model
        dense_model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        target_crs = np.linspace(0.5, 1, NUM_PRUNING_RATIOS)
        for cr in target_crs:
            self.run_test(cr, dense_model)

    def run_test(self, cr, dense_model):
        """
        Runs a pruning test on a pre-trained model with a specified compression rate (cr).

        Args:
            cr (float): The target compression rate (ratio of remaining parameters).
            dense_model (Model): The pre-trained Pytorch model to be pruned.
            test_retraining (bool): If True, retrain the pruned model on dummy data to test stability.

        This function calculates the number of parameters in the dense model, performs pruning to achieve
        the desired compression rate, and validates the actual compression rate achieved. It also tests
        if the outputs of the pruned model are similar to the dense model, and ensures that pruned layers
        respect the importance scores. If `test_retraining` is True, it further validates the model's
        performance after retraining.
        """
        # Calculate the number of parameters in the dense model.
        dense_nparams = count_model_prunable_params(dense_model)

        # Perform pruning on the dense model.
        pruned_model, pruning_info = mct.pruning.pytorch_pruning_experimental(
            model=dense_model,
            target_kpi=mct.KPI(weights_memory=dense_nparams * FP32_BYTES_PER_PARAMETER * cr),
            representative_data_gen=self.representative_dataset,
            pruning_config=mct.pruning.PruningConfig(
                num_score_approximations=1,
                importance_metric=TestImportanceMetric.RANDOM)
        )

        # Calculate the actual compression rate achieved after pruning.
        pruned_nparams = count_model_prunable_params(pruned_model)
        actual_cr = pruned_nparams / dense_nparams
        print(f"Target remaining cr: {cr * 100}, Actual remaining cr: {actual_cr * 100}")

        input_tensor = next(self.representative_dataset())[0]
        pruned_outputs = pruned_model(to_torch_tensor(input_tensor))

        # Ensure pruned layers had lower importance scores than the channels
        # that remained.
        for layer_name, layer_mask in pruning_info.pruning_masks.items():
            if 0 in layer_mask:
                layer_scores = pruning_info.importance_scores[layer_name]
                min_score_remained = min(layer_scores[layer_mask.astype("bool")])
                max_score_removed = max(layer_scores[(1 - layer_mask).astype("bool")])
                self.assertTrue(max_score_removed <= min_score_remained,
                                f"Expected remaining channels to have higher scores"
                                f"than pruned channels but found remained channel with score"
                                f"{min_score_remained} and found pruned channel with"
                                f"score {max_score_removed}")

        # Validate that the actual compression rate does not exceed the target compression rate.
        self.assertTrue(actual_cr <= cr,
                        f"Expected the actual compression rate: {actual_cr} to not exceed the target compression "
                        f"rate: {cr}")


# Function to generate an infinite stream of dummy images and labels
def dummy_data_generator():
    image = np.random.random((3, 224, 224)).astype(np.float32)
    label = np.random.randint(0, 2)
    yield image, label
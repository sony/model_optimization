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

import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

import model_compression_toolkit as mct
import numpy as np
from packaging import version
from model_compression_toolkit.core.pytorch.utils import torch_tensor_to_numpy
from model_compression_toolkit.constants import FP32_BYTES_PER_PARAMETER
from model_compression_toolkit.core.common.pruning.importance_metrics.importance_metric_factory import IMPORTANCE_METRIC_DICT
from model_compression_toolkit.core.pytorch.pytorch_device_config import get_working_device
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor
import torchvision.models as models
from tests.common_tests.pruning.random_importance_metric import RandomImportanceMetric
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
            self.run_test(cr, dense_model, test_retraining=True)

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

    def _dummy_retrain(self, model, trainloader):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9, weight_decay=0.0001)
        model.train()
        device = get_working_device()

        train_loss = 0
        for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader)):
            with torch.cuda.device(device):
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets.squeeze(1))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

        return model

    def run_test(self, cr, dense_model, test_retraining=False):
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

        # Optionally, retrain the pruned model (using dummy data for 1 epoch) and check it
        # predicts differently than before retraining.
        if test_retraining:
            trainloader = create_dummy_trainloader()
            retrained_model = self._dummy_retrain(pruned_model, trainloader)
            retrained_outputs = retrained_model(input_tensor)
            self.assertTrue(np.sum(np.abs(torch_tensor_to_numpy(pruned_outputs) - torch_tensor_to_numpy(retrained_outputs))) != 0, f"Expected after retraining to have different predictions but are the same")

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

class RandomDataset(Dataset):
    def __init__(self, length):
        """
        Initialize the dataset with the specified length.
        :param length: The total number of items in the dataset.
        """
        self.length = length

    def __len__(self):
        """
        Return the total number of items in the dataset.
        """
        return self.length

    def __getitem__(self, index):
        """
        Generate and return a random tensor of size (3, 224, 224).
        :param index: The index of the item (unused, as data is randomly generated).
        :return: A random tensor of size (3, 224, 224).
        """
        # Generate a random tensor with values in the range [0, 1)
        random_tensor = torch.rand(3, 224, 224)
        random_label = torch.randint(0, 1, [1])
        return random_tensor, random_label


# Function to generate an infinite stream of dummy images and labels
def create_dummy_trainloader():
    # Set the desired length (number of data points) for the dataset
    dataset_length = 1000  # 1000 data points

    # Create an instance of the RandomDataset
    random_dataset = RandomDataset(dataset_length)

    # Create a DataLoader to iterate over the RandomDataset
    batch_size = 4  # Define the batch size
    return DataLoader(random_dataset, batch_size=batch_size, shuffle=True)

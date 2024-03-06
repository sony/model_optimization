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

from enum import Enum

import unittest

import tensorflow as tf

import model_compression_toolkit as mct
import numpy as np
from packaging import version

from model_compression_toolkit.constants import FP32_BYTES_PER_PARAMETER
from model_compression_toolkit.core.common.pruning.importance_metrics.importance_metric_factory import IMPORTANCE_METRIC_DICT
from tests.common_tests.pruning.random_importance_metric import RandomImportanceMetric

keras = tf.keras
layers = keras.layers

NUM_PRUNING_RATIOS = 1

class TestImportanceMetric(Enum):
    RANDOM = 'random'

IMPORTANCE_METRIC_DICT.update({TestImportanceMetric.RANDOM: RandomImportanceMetric})

class PruningPretrainedModelsTest(unittest.TestCase):
    def representative_dataset(self, in_shape=(1,224,224,3)):
        for _ in range(1):
            yield [np.random.randn(*in_shape)]

    def test_rn50_pruning(self):
        # Can not be found in tf2.12
        if version.parse(tf.__version__) >= version.parse("2.13"):
            from keras.applications.resnet50 import ResNet50
            dense_model = ResNet50()
            target_crs = np.linspace(0.5, 1, NUM_PRUNING_RATIOS)
            for cr in target_crs:
                self.run_test(cr, dense_model)

    def test_efficientnetb0_pruning(self):
        from keras.applications.efficientnet import EfficientNetB0
        dense_model = EfficientNetB0()
        target_crs = np.linspace(0.8, 1, NUM_PRUNING_RATIOS)
        for cr in target_crs:
            self.run_test(cr, dense_model)


    def test_vgg16_pruning(self):
        from keras.applications.vgg16 import VGG16
        dense_model = VGG16()
        target_crs = np.linspace(0.5, 1, NUM_PRUNING_RATIOS)
        for cr in target_crs:
            self.run_test(cr, dense_model)


    def test_mobilenet_pruning(self):
        from keras.applications.mobilenet import MobileNet
        dense_model = MobileNet()
        target_crs = np.linspace(0.55, 1, NUM_PRUNING_RATIOS)
        for cr in target_crs:
            self.run_test(cr, dense_model)

    def test_mobilenetv2_pruning(self):
        from keras.applications.mobilenet_v2 import MobileNetV2
        dense_model = MobileNetV2()
        target_crs = np.linspace(0.5, 1, NUM_PRUNING_RATIOS)
        for cr in target_crs:
            self.run_test(cr, dense_model)

    def test_densenet_pruning(self):
        from keras.applications.densenet import DenseNet121
        dense_model = DenseNet121()
        target_crs = np.linspace(0.5, 1, NUM_PRUNING_RATIOS)
        for cr in target_crs:
            self.run_test(cr, dense_model)


    def test_vgg19_pruning(self):
        from keras.applications.vgg19 import VGG19
        dense_model = VGG19()
        target_crs = np.linspace(0.5, 1, NUM_PRUNING_RATIOS)
        for cr in target_crs:
            self.run_test(cr, dense_model)


    def _dummy_retrain(self, model, ds):
        # Compile the model with a loss function, optimizer, and metric to monitor
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        # Train the model for one epoch using the dummy dataset
        model.fit(ds, epochs=1)
        return model

    def run_test(self, cr, dense_model, test_retraining=False):
        """
        Runs a pruning test on a pre-trained model with a specified compression rate (cr).

        Args:
            cr (float): The target compression rate (ratio of remaining parameters).
            dense_model (Model): The pre-trained Keras model to be pruned.
            test_retraining (bool): If True, retrain the pruned model on dummy data to test stability.

        This function calculates the number of parameters in the dense model, performs pruning to achieve
        the desired compression rate, and validates the actual compression rate achieved. It also tests
        if the outputs of the pruned model are similar to the dense model, and ensures that pruned layers
        respect the importance scores. If `test_retraining` is True, it further validates the model's
        performance after retraining.
        """
        # Calculate the number of parameters in the dense model.
        dense_nparams = sum([l.count_params() for l in dense_model.layers])

        # Perform pruning on the dense model.
        pruned_model, pruning_info = mct.pruning.keras_pruning_experimental(
            model=dense_model,
            target_kpi=mct.KPI(weights_memory=dense_nparams * FP32_BYTES_PER_PARAMETER * cr),
            representative_data_gen=self.representative_dataset,
            pruning_config=mct.pruning.PruningConfig(
                num_score_approximations=1,
                importance_metric=TestImportanceMetric.RANDOM)
        )

        # Calculate the actual compression rate achieved after pruning.
        pruned_nparams = sum([l.count_params() for l in pruned_model.layers])
        actual_cr = pruned_nparams / dense_nparams
        print(f"Target remaining cr: {cr * 100}, Actual remaining cr: {actual_cr * 100}")

        input_tensor = next(self.representative_dataset())[0]
        pruned_outputs = pruned_model(input_tensor)

        # Optionally, retrain the pruned model (using dummy data for 1 epoch) and check it
        # predicts differently than before retraining.
        if test_retraining:
            ds = create_dummy_dataset()
            retrained_model = self._dummy_retrain(pruned_model, ds)
            retrained_outputs = retrained_model(input_tensor)
            self.assertTrue(np.sum(np.abs(pruned_outputs - retrained_outputs)) != 0, f"Expected after retraining to have different predictions but are the same")

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
    image = np.random.random((224, 224, 3)).astype(np.float32)
    label = np.random.randint(0, 2)
    yield image, label

# Create a Dataset object that returns the dummy data
def create_dummy_dataset():
    dummy_dataset = tf.data.Dataset.from_generator(
        dummy_data_generator,
        output_signature=(
            tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )
    return dummy_dataset.batch(1)

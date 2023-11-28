import unittest

import tensorflow as tf

import model_compression_toolkit as mct
import numpy as np

keras = tf.keras
layers = keras.layers


class ModelsPruningTest(unittest.TestCase):
    def representative_dataset(self, in_shape=(1,224,224,3)):
        for _ in range(1):
            yield [np.random.randn(*in_shape)]

    def test_rn50_pruning(self):
        from keras.applications.resnet50 import ResNet50
        dense_model = ResNet50()
        target_crs = np.linspace(0.5, 1, 5)
        for cr in target_crs:
            self.run_test(cr, dense_model)

    def test_efficientnetb0_pruning(self):
        from keras.applications.efficientnet import EfficientNetB0
        dense_model = EfficientNetB0()
        target_crs = np.linspace(0.6, 1, 5)
        for cr in target_crs:
            self.run_test(cr, dense_model)
    def test_vgg16_pruning(self):
        from keras.applications.vgg16 import VGG16
        dense_model = VGG16()
        target_crs = np.linspace(0.5, 1, 5)
        for cr in target_crs:
            self.run_test(cr, dense_model)


    def test_mobilenet_pruning(self):
        from keras.applications.mobilenet import MobileNet
        dense_model = MobileNet()
        target_crs = np.linspace(0.55, 1, 5)
        for cr in target_crs:
            self.run_test(cr, dense_model)

    def test_mobilenetv2_pruning(self):
        from keras.applications.mobilenet_v2 import MobileNetV2
        dense_model = MobileNetV2()
        target_crs = np.linspace(0.5, 1, 5)
        for cr in target_crs:
            self.run_test(cr, dense_model)

    def test_densenet_pruning(self):
        from keras.applications.densenet import DenseNet121
        dense_model = DenseNet121()
        target_crs = np.linspace(0.5, 1, 5)
        for cr in target_crs:
            self.run_test(cr, dense_model)


    def test_vgg19_pruning(self):
        from keras.applications.vgg19 import VGG19
        dense_model = VGG19()
        target_crs = np.linspace(0.5, 1, 5)
        for cr in target_crs:
            self.run_test(cr, dense_model)


    def run_test(self, cr, dense_model):
        dense_nparams = sum([l.count_params() for l in dense_model.layers])
        pruned_model, pruning_info = mct.pruning.keras_pruning_experimental(model=dense_model,
                                                                            target_kpi=mct.KPI(
                                                                                weights_memory=dense_nparams * 4. * cr),
                                                                            representative_data_gen=self.representative_dataset,
                                                                            pruning_config=mct.pruning.PruningConfig(
                                                                                num_score_approximations=1,
                                                                            importance_metric=mct.pruning.ImportanceMetric.RANDOM))
        pruned_nparams = sum([l.count_params() for l in pruned_model.layers])
        actual_cr = pruned_nparams / dense_nparams
        print(f"Target cr: {cr}, Actual cr: {actual_cr}")
        input_tensor = next(self.representative_dataset())[0]
        pruned_outputs = pruned_model(input_tensor)
        # TODO: test dummy retraining

        for layer_name, layer_mask in pruning_info.pruning_masks.items():
            if 0 in layer_mask:
                layer_scores = pruning_info.importance_scores[layer_name]
                min_score_remained = min(layer_scores[layer_mask.astype("bool")])
                max_score_removed = max(layer_scores[(1-layer_mask).astype("bool")])
                assert max_score_removed <= min_score_remained

        assert actual_cr <= cr
        # print(f"Target cr: {cr}, Actual cr: {actual_cr}")
        if cr>=1.:
            assert np.sum(np.abs(pruned_outputs-dense_model(input_tensor)))==0


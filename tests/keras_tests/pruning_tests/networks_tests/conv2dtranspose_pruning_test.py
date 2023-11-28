
import tensorflow as tf

import model_compression_toolkit as mct
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import generate_keras_tpc
from tests.common_tests.helpers.generate_test_tp_model import generate_test_tp_model

from tests.keras_tests.pruning_tests.pruning_keras_feature_test import PruningKerasFeatureTest

keras = tf.keras
layers = keras.layers


class Conv2DTransposePruningTest(PruningKerasFeatureTest):
    """
    Test a network with two adjacent dense and check it's pruned for a target compression ratio.
    """

    def __init__(self,
                 unit_test,
                 target_cr=0.5,
                 use_bn=False,
                 activation_layer=None):
        super().__init__(unit_test,
                         input_shape=(8, 8, 3))
        self.target_cr = target_cr
        self.use_bn = use_bn
        self.activation_layer = activation_layer

    def get_tpc(self):
        tp = generate_test_tp_model({'simd_size': 1})
        return generate_keras_tpc(name="simd2_test", tp_model=tp)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2DTranspose(filters=3, kernel_size=1)(inputs)
        if self.use_bn:
            x = layers.BatchNormalization()(x)
        if self.activation_layer:
            x = self.activation_layer(x)
        x = layers.Conv2DTranspose(filters=4, kernel_size=1)(x)
        x = layers.Conv2DTranspose(filters=4, kernel_size=1)(x)
        model = keras.Model(inputs=inputs, outputs=x)
        return model

    def get_kpi(self):
        return mct.KPI(weights_memory=self.dense_model_num_params * 4 * self.target_cr)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        pruned_nparams = sum([l.count_params() for l in quantized_model.layers])
        remaining_cr = pruned_nparams / self.dense_model_num_params
        self.unit_test.assertTrue(remaining_cr <= self.target_cr)
        print(f"Remaining CR: {remaining_cr}")
        if self.target_cr>=1:
            self.unit_test.assertTrue(remaining_cr==1)

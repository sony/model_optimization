from typing import List

from tests.common_tests.base_test import BaseTest, TestMode


class BaseLayerTest(BaseTest):
    def __init__(self,
                 unit_test,
                 val_batch_size=1,
                 num_calibration_iter=1,
                 num_of_inputs=1,
                 input_shape=(8, 8, 3),
                 quantization_modes: List[TestMode] = [TestMode.QUANTIZED_16_BITS, TestMode.FLOAT],
                 is_inputs_a_list=False):

        super().__init__(unit_test,
                         val_batch_size=val_batch_size,
                         num_calibration_iter=num_calibration_iter,
                         num_of_inputs=num_of_inputs,
                         input_shape=input_shape,
                         quantization_modes=quantization_modes)

        self.is_inputs_a_list = is_inputs_a_list

    def get_layers(self):
        raise Exception('Implement it')

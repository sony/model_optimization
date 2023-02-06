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
from abc import abstractmethod

import numpy as np
from model_compression_toolkit.core.common import Logger
from model_compression_toolkit.core.common.target_platform import QuantizationMethod
from model_compression_toolkit.quantizers_infrastructure.common.base_inferable_quantizer import mark_quantizer


from model_compression_toolkit.core.common.constants import FOUND_TF
from model_compression_toolkit.quantizers_infrastructure.common.quant_utils import adjust_range_to_include_zero
from model_compression_toolkit.quantizers_infrastructure.common.base_inferable_quantizer import QuantizationTarget


if FOUND_TF:
    import tensorflow as tf
    from model_compression_toolkit.quantizers_infrastructure.keras.inferable_quantizers.base_keras_inferable_quantizer \
        import \
        BaseKerasInferableQuantizer

    @mark_quantizer(quantization_target=None,
                    quantization_method=[QuantizationMethod.UNIFORM],
                    quantizer_type=None)
    class BaseUniformInferableQuantizer(BaseKerasInferableQuantizer):

        def __init__(self,
                     num_bits: int,
                     min_range: np.ndarray,
                     max_range: np.ndarray,
                     ):
            """
            Initialize the quantizer with the specified parameters.

            Args:
                num_bits: number of bits to use for quantization
                min_range: min quantization range
                max_range: max quantization range
            """
            super(BaseUniformInferableQuantizer, self).__init__()

            assert isinstance(min_range, list), f'Expected min_range to be of type list but is {type(min_range)}'
            assert isinstance(max_range, list), f'Expected max_range to be of type list but is {type(max_range)}'

            assert all([isinstance(x, (float, np.float32)) for x in
                        min_range]), f'Expected min_range list to contain float values but found ' \
                                     f'{[type(x) for x in min_range]}'
            assert all([isinstance(x, (float, np.float32)) for x in
                        max_range]), f'Expected max_range list to contain float values but found ' \
                                     f'{[type(x) for x in max_range]}'

            assert len(min_range) == len(max_range), f'Expected min/max values to have the same length but min shape: {len(min_range)} and max shape: {len(max_range)}'

            self.num_bits = num_bits

            # Convert min/max to numpy arrays
            min_range, max_range = np.asarray(min_range), np.asarray(max_range)

            assert np.all(max_range > min_range), f'Expected max_range to be bigger than min_range!'
            _min_range, _max_range = adjust_range_to_include_zero(min_range, max_range, num_bits)
            assert np.all(_min_range <= 0) and np.all(_max_range >= 0), f'Expected zero to be in the range, got min_range={_min_range}, max_range={_max_range}'
            if not np.isclose(np.linalg.norm(_min_range-min_range),0,atol=1e-6) or not np.isclose(np.linalg.norm(_max_range-max_range),0,atol=1e-6):
                Logger.warning(f"Adjusting (min_range, max_range) from ({min_range},{max_range}) to ({_min_range},{_max_range})")  # pragma: no cover

            self.max_range = _max_range
            self.min_range = _min_range

        @abstractmethod
        def get_config(self):
            """
            Return a dictionary with the configuration of the quantizer.
            """
            raise NotImplemented(f'{self.__class__.__name__} did not implement get_config')  # pragma: no cover

else:
    class BaseUniformInferableQuantizer:
        def __init__(self, *args, **kwargs):
            raise Exception('Installing tensorflow and tensorflow_model_optimization is mandatory '
                            'when using BaseUniformInferableQuantizer. '
                            'Could not find Tensorflow package.')

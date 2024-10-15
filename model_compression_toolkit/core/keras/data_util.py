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
from typing import Generator, Callable

import tensorflow as tf

from model_compression_toolkit.core.keras.tf_tensor_numpy import to_tf_tensor


def flat_gen_fn(data_gen_fn: Callable[[], Generator]):
    """
    Convert data generator with arbitrary batch size to a flat (sample by sample) data generator.

    Args:
        data_gen_fn: input data generator factory. Generator is expected to yield lists of tensors.

    Returns:
        A factory for a flattened data generator.
    """
    def gen():
        for inputs_batch in data_gen_fn():
            for sample in zip(*inputs_batch):
                yield to_tf_tensor(sample)
    return gen


# TODO in tf dataset and dataloader are combined within tf.data.Dataset. For advanced use cases such as gptq sla we
#  need to separate dataset from dataloader similarly to torch data_util.
class TFDatasetFromGenerator:
    def __init__(self, data_gen, batch_size):
        inputs = next(data_gen())
        if not isinstance(inputs, list):
            raise TypeError(f'Representative data generator is expected to generate a list of tensors, '
                            f'got {type(inputs)}')  # pragma: no cover

        self.orig_batch_size = inputs[0].shape[0]

        output_signature = tuple([tf.TensorSpec(shape=t.shape[1:], dtype=t.dtype) for t in inputs])
        dataset = tf.data.Dataset.from_generator(flat_gen_fn(data_gen), output_signature=output_signature)
        self.dataset = dataset.batch(batch_size)
        self._size = None

    def __iter__(self):
        return iter(self.dataset)

    def __len__(self):
        """ Returns the number of batches. """
        if self._size is None:
            self._num_batches = sum(1 for _ in self)
        return self._num_batches


def data_gen_to_dataloader(data_gen_fn: Callable[[], Generator], batch_size) -> TFDatasetFromGenerator:
    """ Create DataLoader based on samples yielded by data_gen. """
    return TFDatasetFromGenerator(data_gen_fn, batch_size)

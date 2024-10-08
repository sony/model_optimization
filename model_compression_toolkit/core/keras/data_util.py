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
import math
from typing import Generator, Callable

import tensorflow as tf


def flat_gen_fn(data_gen_fn: Callable[[], Generator]):
    def gen():
        for inputs_batch in data_gen_fn():
            for i in range(inputs_batch[0].shape[0]):
                yield tuple(t[i] for t in inputs_batch)
    return gen


class TFDataLoader:
    def __init__(self, data_gen, batch_size, num_samples):
        inputs = next(data_gen())
        if not isinstance(inputs, list):
            raise TypeError(f'Representative data generator is expected to generate a list of tensors, '
                            f'got {type(inputs)}')  # pragma: no cover

        output_signature = tuple([tf.TensorSpec(shape=t.shape[1:]) for t in inputs])
        dataset = tf.data.Dataset.from_generator(flat_gen_fn(data_gen), output_signature=output_signature)
        if num_samples:
            dataset = dataset.take(num_samples)
        dataset = dataset.batch(batch_size)
        self.dataset = dataset
        self.batch_size = batch_size
        self._num_batches = None

    def __iter__(self):
        return iter(self.dataset)

    def __len__(self):
        if self._num_batches is None:
            self._num_batches = 0
            for _ in self.dataset:
                self._num_batches += 1

        return self._num_batches


def create_dataloader_from_data_generator(data_gen: Callable[[], Generator],
                                          batch_size: int, num_samples: int = None) -> TFDataLoader:
    """
    Creates a data loader from representative dataset generator.

    Args:
        data_gen: original data generator (of any batch size).
        batch_size: batch size for the new dataset.
        num_samples: use first num_samples for the new dataset. If None, all samples are retained.

    Returns:
        Data loader.
    """
    return TFDataLoader(data_gen, batch_size, num_samples)

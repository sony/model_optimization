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

import tensorflow as tf
from typing import Callable, Generator, Sequence, Any


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
                yield [tf.convert_to_tensor(s) for s in sample]

    return gen


class TFDatasetFromGenerator:
    """
    TensorFlow dataset from a data generator function, batched to a specified size.
    """

    def __init__(self, data_gen_fn: Callable[[], Generator], batch_size: int):
        """
        Args:
            data_gen_fn: a factory function for data generator that yields lists of tensors.
            batch_size: the batch size for the dataset.
        """
        inputs = next(data_gen_fn())
        if not isinstance(inputs, list):
            raise TypeError(f'Data generator is expected to yield a list of tensors, got {type(inputs)}')
        self.orig_batch_size = inputs[0].shape[0]

        output_signature = tuple([tf.TensorSpec(shape=t.shape[1:], dtype=t.dtype) for t in inputs])
        self.dataset = tf.data.Dataset.from_generator(flat_gen_fn(data_gen_fn), output_signature=output_signature)
        self.dataset = self.dataset.batch(batch_size)

    def __iter__(self):
        return iter(self.dataset)

    def __len__(self):
        """ Returns the number of batches. """
        if self._size is None:
            self._size = sum(1 for _ in self.dataset)
        return self._size


class FixedTFDataset:
    """
    Fixed dataset containing samples from a generator, stored in memory.
    """

    def __init__(self, data_gen_fn: Callable[[], Generator], n_samples: int = None):
        """
        Args:
            data_gen_fn: data generator function.
            n_samples: number of samples to store in the dataset. If None, uses all samples in one pass.
        """
        inputs = next(data_gen_fn())
        if not isinstance(inputs, list):
            raise TypeError(f'Data generator is expected to yield a list of tensors, got {type(inputs)}')
        self.orig_batch_size = inputs[0].shape[0]

        samples = []
        for batch in data_gen_fn():
            samples.extend(zip(*[tf.convert_to_tensor(t) for t in batch]))
            if n_samples is not None and len(samples) >= n_samples:
                samples = samples[:n_samples]
                break

        if n_samples and len(samples) < n_samples:
            raise ValueError(f'Not enough samples to create a dataset with {n_samples} samples')
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class FixedSampleInfoDataset:
    """
    Dataset for samples with additional info, each element is a tuple of (sample, sample_info).
    """

    def __init__(self, samples: Sequence, *sample_info: Sequence):
        if not all(len(info) == len(samples) for info in sample_info):
            raise ValueError('Sample and additional info lengths must match')
        self.samples = samples
        self.sample_info = sample_info

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index], *[info[index] for info in self.sample_info]


class IterableSampleWithConstInfoDataset:
    """
    Augments each sample in an iterable dataset with constant additional information.
    """

    def __init__(self, samples_dataset: tf.data.Dataset, *info: Any):
        self.samples_dataset = samples_dataset
        self.info = info

    def __iter__(self):
        for sample in self.samples_dataset:
            yield (sample, *self.info)


def data_gen_to_dataloader(data_gen_fn: Callable[[], Generator], batch_size: int) -> TFDatasetFromGenerator:
    """Create a DataLoader based on samples yielded by data_gen."""
    return TFDatasetFromGenerator(data_gen_fn, batch_size)


def create_tf_dataloader(dataset, batch_size=7, shuffle=False, collate_fn=None, extra_output=None):
    """
    Creates a tf.data.Dataset with specified loading options.

    Args:
        dataset: The dataset container (e.g., FixedDatasetFromGenerator or FixedSampleInfoDataset).
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the dataset.
        collate_fn: A function to apply to each batch (e.g., add extra outputs like regularization weights).
        extra_output: Tensor to add as an extra output to each batch, if needed.

    Returns:
        tf.data.Dataset: Configured for batching, shuffling, and custom transformations.
    """
    def generator():
        for i in range(len(dataset)):
            yield dataset[i]

    # Create tf.data.Dataset from generator
    dummy_input_tensors = next(generator())
    output_signature = tuple([tf.TensorSpec(shape=t.shape, dtype=t.dtype) for t in dummy_input_tensors])
    tf_dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature
    )

    if shuffle:
        tf_dataset = tf_dataset.shuffle(buffer_size=len(dataset))

    tf_dataset = tf_dataset.batch(batch_size)

    # Apply collate function if provided
    if collate_fn:
        tf_dataset = tf_dataset.map(lambda *args: collate_fn(*args, extra_output=extra_output))

    return tf_dataset

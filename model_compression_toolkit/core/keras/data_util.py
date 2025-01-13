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

import tensorflow as tf
from typing import Callable, Generator, Sequence, Any


def get_tensor_spec(item, ignore_batch_dim=False):
    """
    Get the TensorFlow TensorSpec for an item, optionally ignoring the first dimension.

    Args:
        item: The input item, which could be a tensor, tuple, or list.
        ignore_batch_dim (bool): Whether to ignore the first dimension of the tensor shape.

    Returns:
        TensorSpec or a tuple of TensorSpecs.
    """
    if isinstance(item, (tuple, list)):
        return tuple(get_tensor_spec(sub_item, ignore_batch_dim) for sub_item in item)

    shape = item.shape[1:] if ignore_batch_dim else item.shape
    return tf.TensorSpec(shape=shape, dtype=item.dtype)


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
                yield tuple([tf.convert_to_tensor(s) for s in sample])

    return gen

class TFDatasetFromGenerator:
    """
    TensorFlow dataset from a data generator function, batched to a specified size.
    """

    def __init__(self, data_gen_fn: Callable[[], Generator]):
        """
        Args:
            data_gen_fn: a factory function for data generator that yields lists of tensors.
        """
        inputs = next(data_gen_fn())
        if not isinstance(inputs, list):
            raise TypeError(f'Data generator is expected to yield a list of tensors, got {type(inputs)}')  # pragma: no cover
        self.orig_batch_size = inputs[0].shape[0]
        self._size = None

        # TFDatasetFromGenerator flattens the dataset, thus we ignore the batch dimension
        output_signature = get_tensor_spec(inputs, ignore_batch_dim=True)
        self.tf_dataset = tf.data.Dataset.from_generator(flat_gen_fn(data_gen_fn), output_signature=output_signature)

    def __iter__(self):
        return iter(self.tf_dataset)

    def __len__(self):
        """ Returns the number of batches. """
        if self._size is None:
            self._size = sum(1 for _ in self.tf_dataset)
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
            raise TypeError(f'Data generator is expected to yield a list of tensors, got {type(inputs)}')  # pragma: no cover
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

        # Use from_generator to keep tuples intact
        self.tf_dataset = tf.data.Dataset.from_generator(
            lambda: iter(self.samples),
            output_signature=tuple(tf.TensorSpec(shape=sample.shape, dtype=sample.dtype) for sample in self.samples[0])
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class FixedSampleInfoDataset:
    """
    Dataset for samples with additional info, each element is a tuple of (sample, sample_info).
    """

    def __init__(self, samples: Sequence, sample_info: Sequence):
        if not all(len(info) == len(samples) for info in sample_info):
            raise ValueError('Sample and additional info lengths must match')  # pragma: no cover
        self.samples = samples
        self.sample_info = sample_info

        # Get the number of tensors in each tuple (corresponds to the number of input layers the model has)
        num_tensors = len(samples[0])

        # Create separate lists: one for each input layer and separate the tuples into lists
        sample_tensor_lists = [[] for _ in range(num_tensors)]
        for s in samples:
            for i, data_tensor in enumerate(s):
                sample_tensor_lists[i].append(data_tensor)

        # In order to deal with models that have different input shapes for different layers, we need first to
        # organize the data in a dictionary in order to use tf.data.Dataset.from_tensor_slices
        samples_dict = {f'tensor_{i}': tensors for i, tensors in enumerate(sample_tensor_lists)}
        info_dict = {f'info_{i}': tf.convert_to_tensor(info) for i, info in enumerate(self.sample_info)}
        combined_dict = {**samples_dict, **info_dict}

        tf_dataset = tf.data.Dataset.from_tensor_slices(combined_dict)

        # Map the dataset to return tuples instead of dict
        def reorganize_ds_outputs(ds_output):
            tensors = tuple(ds_output[f'tensor_{i}'] for i in range(num_tensors))
            infos = tuple(ds_output[f'info_{i}'] for i in range(len(sample_info)))
            return tensors, infos

        self.tf_dataset = tf_dataset.map(reorganize_ds_outputs)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index], tuple([info[index] for info in self.sample_info])


class IterableSampleWithConstInfoDataset:
    """
    Augments each sample in an iterable dataset with constant additional information.
    """

    def __init__(self, samples_dataset: tf.data.Dataset, *info: Any):
        self.samples_dataset = samples_dataset
        self.info = info

        # Map to ensure the output is always (sample, info) as a tuple
        self.tf_dataset = self.samples_dataset.map(
            lambda *x: ((x,) if not isinstance(x, tuple) else x, *self.info)
        )

    def __iter__(self):
        for sample in self.samples_dataset:
            yield ((sample,) if not isinstance(sample, tuple) else sample, *self.info)


def data_gen_to_dataloader(data_gen_fn: Callable[[], Generator], batch_size: int):
    """Create a DataLoader based on samples yielded by data_gen."""
    ds = TFDatasetFromGenerator(data_gen_fn)
    return create_tf_dataloader(mct_dataset=ds, batch_size=batch_size)


def create_tf_dataloader(mct_dataset, batch_size, shuffle=False, collate_fn=None):
    """
    Creates a tf.data.Dataset with specified loading options.

    Args:
        dataset: The dataset container (e.g., FixedDatasetFromGenerator or FixedSampleInfoDataset).
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the dataset.
        collate_fn: A function to apply to each batch (e.g., add extra outputs like regularization weights).

    Returns:
        tf.data.Dataset: Configured for batching, shuffling, and custom transformations.
    """
    dataset = mct_dataset.tf_dataset

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(dataset))

    dataset = dataset.batch(batch_size)

    # Apply collate function if provided
    if collate_fn:
        dataset = dataset.map(lambda *args: collate_fn(args))

    return dataset

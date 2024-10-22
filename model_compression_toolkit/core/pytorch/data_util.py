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
from typing import Generator, Callable, Sequence, Any

import torch
from torch.utils.data import IterableDataset, Dataset, DataLoader, default_collate


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
                # convert to torch tensor but do not move to device yet (it will cause issues with num_workers > 0)
                yield [torch.as_tensor(s) for s in sample]
    return gen


class IterableDatasetFromGenerator(IterableDataset):
    """
    PyTorch iterable dataset built from a data generator factory.
    Each iteration over the dataset corresponds to one pass over a fresh instance of a data generator.
    Therefore, if the data generator factory creates data generator instances that yield different samples,
    this behavior is preserved.
    """

    def __init__(self, data_gen_fn: Callable[[], Generator]):
        """
        Args:
            data_gen_fn: a factory for data generator that yields lists of tensors.
        """
        # validate one batch
        test_batch = next(data_gen_fn())
        if not isinstance(test_batch, list):
            raise TypeError(f'Data generator is expected to yield a list of tensors, got {type(test_batch)}')
        self.orig_batch_size = test_batch[0].shape[0]

        self._size = None
        self._gen_fn = flat_gen_fn(data_gen_fn)

    def __iter__(self):
        """ Return an iterator for the dataset. """
        return self._gen_fn()

    def __len__(self):
        """ Get the length of the dataset. """
        if self._size is None:
            self._size = sum(1 for _ in self)
        return self._size


class FixedDatasetFromGenerator(Dataset):
    """
    Dataset containing a fixed number of samples (i.e. same samples are yielded in each epoch), retrieved from a
    data generator.
    Note that the samples are stored in memory.

    Attributes:
        orig_batch_size: the batch size of the input data generator (retrieved from the first batch).
    """
    def __init__(self, data_gen_fn: Callable[[], Generator], n_samples: int = None):
        """
        Args:
            data_gen_fn: data generator factory.
            n_samples: target size of the dataset. If None, use all samples yielded by the data generator in one pass.
        """
        test_batch = next(data_gen_fn())
        if not isinstance(test_batch, list):
            raise TypeError(f'Data generator is expected to yield a list of tensors, got {type(test_batch)}')
        self.orig_batch_size = test_batch[0].shape[0]

        samples = []
        for batch in data_gen_fn():
            # convert to torch tensor but do not move to device yet (it will cause issues with num_workers > 0)
            batch = [torch.as_tensor(t) for t in batch]
            samples.extend(zip(*batch))
            if n_samples is not None and len(samples) >= n_samples:
                samples = samples[:n_samples]
                break

        if n_samples is not None and len(samples) < n_samples:
            raise ValueError(f'Not enough samples in the data generator to create a dataset with {n_samples}')
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return list(self.samples[index])


class FixedSampleInfoDataset(Dataset):
    """
    Dataset for samples augmented with additional info per sample.
    Each element in the dataset is a tuple containing the sample and sample's additional info.
    """
    def __init__(self, samples: Sequence, *sample_info: Sequence):
        """
        Args:
            samples: a sequence of input samples.
            hessians: one or more sequences of samples complementary data of matching sizes.
        """
        if not all(len(info) == len(samples) for info in sample_info):
            raise ValueError('Mismatch in the number of samples between samples and complementary data.')
        self.samples = samples
        self.sample_info = sample_info

    def __getitem__(self, index):
        return self.samples[index], *[info[index] for info in self.sample_info]

    def __len__(self):
        return len(self.samples)


class IterableSampleWithConstInfoDataset(IterableDataset):
    """
    A convenience dataset that augments each sample with additional info shared by all samples.
    """
    def __init__(self, samples_dataset: Dataset, *info: Any):
        """
        Args:
            samples_dataset: any dataset containing samples.
            *sample_info: one or more static entities to augment each sample.
        """
        self.samples_dataset = samples_dataset
        self.info = info

    def __iter__(self):
        for sample in self.samples_dataset:
            yield sample, *self.info


def get_collate_fn_with_extra_outputs(*extra_outputs: Any) -> Callable:
    """ Collation function that adds const extra outputs to each batch. """
    def f(batch):
        return default_collate(batch) + list(extra_outputs)
    return f


def data_gen_to_dataloader(data_gen_fn: Callable[[], Generator], batch_size, **kwargs):
    """ Create DataLoader based on samples yielded by data_gen. """
    dataset = IterableDatasetFromGenerator(data_gen_fn)
    return DataLoader(dataset, batch_size=batch_size, **kwargs)

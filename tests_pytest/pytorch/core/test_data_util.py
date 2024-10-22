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
import pytest
import torch
import numpy as np
from torch.utils.data import IterableDataset, Dataset

from model_compression_toolkit.core.pytorch.data_util import (data_gen_to_dataloader, IterableDatasetFromGenerator,
                                                              FixedDatasetFromGenerator, FixedSampleInfoDataset)


@pytest.fixture(scope='session')
def fixed_dataset():
    # generate 320 images with data1[i] = i and data2[i] = i+10
    data1 = np.stack([np.full((3, 30, 20), v) for v in range(320)], axis=0)
    data2 = np.stack([np.full((10,), v + 10) for v in range(320)], axis=0)
    return data1, data2


@pytest.fixture
def fixed_gen(fixed_dataset):
    def f():
        for i in range(10):
            yield [fixed_dataset[0][32 * i: 32 * (i + 1)], fixed_dataset[1][32 * i: 32 * (i + 1)]]

    return f


def get_random_data_gen_fn(seed=42):
    """ get gen factory for reproducible gen yielding different samples in each epoch """
    rng = np.random.default_rng(seed)

    def f():
        for i in range(10):
            yield [rng.random((32, 3, 20, 30)), rng.random((32, 10))]
    return f


class TestDataUtil:
    create_dataloader_fn = data_gen_to_dataloader

    def test_iterable_dataset_from_fixed_gen(self, fixed_gen):
        """ tests iterable dataset from fixed gen - same samples are generated in each epoch in the same order """
        ds = IterableDatasetFromGenerator(fixed_gen)
        assert isinstance(ds, IterableDataset)
        self._validate_ds_from_fixed_gen(ds, 320)

    def test_iterable_dataset_from_random_gen(self):
        """ test that dataset samples over epochs are identical to the original data generator """
        ds = IterableDatasetFromGenerator(get_random_data_gen_fn())
        pass1 = torch.stack([t[0] for t in ds], dim=0)
        pass2 = torch.stack([t[0] for t in ds], dim=0)

        gen_fn = get_random_data_gen_fn()
        # one invocation is used for validation and batch size in dataset, so promote the reference gen for comparison
        next(gen_fn())
        gen_pass1 = np.concatenate([t[0] for t in gen_fn()], axis=0)
        gen_pass2 = np.concatenate([t[0] for t in gen_fn()], axis=0)
        # check that each pass is identical to corresponding pass in the original gen
        assert np.allclose(pass1.cpu().numpy(), gen_pass1)
        assert np.allclose(pass2.cpu().numpy(), gen_pass2)
        assert not torch.equal(pass1, pass2)

    def test_fixed_dataset_from_fixed_gen_full(self, fixed_gen):
        ds = FixedDatasetFromGenerator(fixed_gen)
        assert isinstance(ds, Dataset) and not isinstance(ds, IterableDataset)
        self._validate_ds_from_fixed_gen(ds, 320)

    def test_fixed_dataset_from_const_gen_subset(self, fixed_gen):
        ds = FixedDatasetFromGenerator(fixed_gen, n_samples=25)
        self._validate_ds_from_fixed_gen(ds, 25)

    def test_fixed_dataset_from_random_gen_full(self):
        ds = FixedDatasetFromGenerator(get_random_data_gen_fn())
        self._validate_fixed_ds(ds, exp_len=320, exp_batch_size=32)

    def test_fixed_dataset_from_random_gen_subset(self):
        ds = FixedDatasetFromGenerator(get_random_data_gen_fn(), n_samples=123)
        self._validate_fixed_ds(ds, exp_len=123, exp_batch_size=32)

    def test_not_enough_samples_in_datagen(self):
        def gen():
            yield [np.ones((10, 3))]
        with pytest.raises(ValueError, match='Not enough samples in the data generator'):
            FixedDatasetFromGenerator(gen, n_samples=11)

    def test_extra_info_mismatch(self, fixed_gen):
        with pytest.raises(ValueError, match='Mismatch in the number of samples between samples and complementary data'):
            FixedSampleInfoDataset([1]*10, [2]*10, [3]*11)

    @pytest.mark.parametrize('ds_cls', [FixedDatasetFromGenerator, IterableDatasetFromGenerator])
    def test_invalid_gen(self, ds_cls):
        def gen():
            yield np.ones((10, 3))
        with pytest.raises(TypeError, match='Data generator is expected to yield a list of tensors'):
            ds_cls(gen)

    def _validate_ds_from_fixed_gen(self, ds, exp_len):
        for _ in range(2):
            for i, sample in enumerate(ds):
                assert np.array_equal(sample[0].cpu().numpy(), np.full((3, 30, 20), i))
                assert np.array_equal(sample[1].cpu().numpy(), np.full((10,), i + 10))
            assert i == exp_len - 1
            assert ds.orig_batch_size == 32
            assert len(ds) == exp_len

    def _validate_fixed_ds(self, ds, exp_len, exp_batch_size):
        assert isinstance(ds, torch.utils.data.Dataset) and not isinstance(ds, torch.utils.data.IterableDataset)
        full_pass1 = torch.concat([t[0] for t in ds], dim=0)
        full_pass2 = torch.concat([t[0] for t in ds], dim=0)
        assert torch.equal(full_pass1, full_pass2)
        assert len(ds) == exp_len
        assert ds.orig_batch_size == exp_batch_size

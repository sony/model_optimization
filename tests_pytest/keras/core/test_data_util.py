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
import numpy as np
import pytest

from model_compression_toolkit.core.keras.data_util import data_gen_to_dataloader, TFDatasetFromGenerator


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
            yield [rng.random((32, 3, 20, 30)).astype(np.float32), rng.random((32, 10)).astype(np.float32)]
    return f


class TestTFDataUtil:
    create_dataloader_fn = data_gen_to_dataloader

    def test_iterable_dataset_from_fixed_gen(self, fixed_gen):
        """ tests iterable dataset from fixed gen - same samples are generated in each epoch in the same order """
        ds = TFDatasetFromGenerator(fixed_gen, batch_size=1)
        self._validate_ds_from_fixed_gen(ds, 320)

    def test_iterable_dataset_from_random_gen(self):
        """ test that dataset samples over epochs are identical to the original data generator """
        ds = TFDatasetFromGenerator(get_random_data_gen_fn(), batch_size=1)
        pass1 = np.concatenate([t[0] for t in ds], axis=0)
        pass2 = np.concatenate([t[0] for t in ds], axis=0)

        gen_fn = get_random_data_gen_fn()
        # one invocation is used for validation and batch size in dataset, so promote the reference gen for comparison
        next(gen_fn())
        gen_pass1 = np.concatenate([t[0] for t in gen_fn()], axis=0)
        gen_pass2 = np.concatenate([t[0] for t in gen_fn()], axis=0)
        # check that each pass is identical to corresponding pass in the original gen
        assert np.array_equal(pass1, gen_pass1)
        assert np.array_equal(pass2, gen_pass2)
        assert not np.allclose(pass1, pass2)

    def test_dataloader(self, fixed_gen):
        ds = TFDatasetFromGenerator(fixed_gen, batch_size=25)
        ds_iter = iter(ds)
        batch1 = next(ds_iter)
        assert batch1[0].shape[0] == batch1[1].shape[0] == 25
        assert np.array_equal(batch1[0][0], np.full((3, 30, 20), 0))
        assert np.array_equal(batch1[1][0], np.full((10,), 10))
        assert np.array_equal(batch1[0][-1], np.full((3, 30, 20), 24))
        assert np.array_equal(batch1[1][-1], np.full((10,), 34))
        assert len(ds) == 13
        assert ds.orig_batch_size == 32

    def _validate_ds_from_fixed_gen(self, ds, exp_len):
        for _ in range(2):
            for i, sample in enumerate(ds):
                assert np.array_equal(sample[0].cpu().numpy(), np.full((1, 3, 30, 20), i))
                assert np.array_equal(sample[1].cpu().numpy(), np.full((1, 10,), i + 10))
            assert i == exp_len - 1
            assert ds.orig_batch_size == 32
            assert len(ds) == exp_len

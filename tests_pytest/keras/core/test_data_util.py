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

from model_compression_toolkit.core.keras.data_util import data_gen_to_dataloader, TFDatasetFromGenerator, create_tf_dataloader, FixedTFDataset, IterableSampleWithConstInfoDataset, FixedSampleInfoDataset

import tensorflow as tf

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

    def test_create_tf_dataloader_tfdataset_from_generator(self, fixed_gen):
        batch_size = 2
        dataloader = data_gen_to_dataloader(fixed_gen, batch_size)
        assert isinstance(dataloader, tf.data.Dataset)
        assert len(list(dataloader)) == 320 // batch_size

        for batch in dataloader:
            inputs, labels = batch
            assert inputs.shape == (batch_size, 3, 30, 20)
            assert labels.shape == (batch_size, 10)

    def test_create_tf_dataloader_fixed_tfdataset(self, fixed_gen):
        n_samples = 64
        dataset = FixedTFDataset(fixed_gen, n_samples=n_samples)

        for i, sample in enumerate(dataset):
            assert np.array_equal(sample[0].cpu().numpy(), np.full((3, 30, 20), i))

        batch_size = 16
        dataloader = create_tf_dataloader(dataset, batch_size=batch_size)

        assert isinstance(dataloader, tf.data.Dataset)
        assert len(list(dataloader)) == n_samples // batch_size

        for batch in dataloader:
            inputs, labels = batch
            assert inputs.shape == (batch_size, 3, 30, 20)
            assert labels.shape == (batch_size, 10)

    def test_fixed_tfdataset_too_many_requested_samples(self, fixed_gen):
        n_samples = 321
        with pytest.raises(Exception) as e_info:
            FixedTFDataset(fixed_gen, n_samples=n_samples)
        assert 'Not enough samples to create a dataset with 321 samples' in str(e_info)

    def test_create_tf_dataloader_fixed_tfdataset_with_info(self, fixed_gen):
        samples = []
        for b in list(fixed_gen()):
            samples.extend(tf.unstack(b[0], axis=0))
            break # Take one batch only (since this tests fixed,small dataset)
        dataset = FixedSampleInfoDataset(samples, [tf.range(32)])

        for i, sample_with_info in enumerate(dataset):
            sample, info = sample_with_info
            assert np.array_equal(sample.cpu().numpy(), np.full((3, 30, 20), i))
            assert info == (i,)

        batch_size = 16
        dataloader = create_tf_dataloader(dataset, batch_size=batch_size)

        assert isinstance(dataloader, tf.data.Dataset)
        assert len(list(dataloader)) == 32 // batch_size

        for batch in dataloader:
            samples, additional_info = batch
            assert samples.shape == (batch_size, 3, 30, 20)
            assert additional_info[0].shape == (batch_size,)

    def test_create_tf_dataloader_iterable_tfdataset_with_const_info(self, fixed_gen):
        iterable_ds = TFDatasetFromGenerator(fixed_gen)
        dataset = IterableSampleWithConstInfoDataset(iterable_ds, tf.constant("some_string"))

        for i, sample_with_info in enumerate(dataset):
            sample, info = sample_with_info
            assert np.array_equal(sample[0].cpu().numpy(), np.full((3, 30, 20), i))
            assert info == tf.constant("some_string")

        batch_size = 16
        dataloader = create_tf_dataloader(dataset, batch_size=batch_size)

        assert isinstance(dataloader, tf.data.Dataset)
        assert len(list(dataloader)) == 320 // batch_size

        for batch in dataloader:
            samples, additional_info = batch
            assert samples[0].shape == (batch_size, 3, 30, 20)
            assert additional_info.shape == (batch_size,)
            assert all(additional_info == tf.constant("some_string"))
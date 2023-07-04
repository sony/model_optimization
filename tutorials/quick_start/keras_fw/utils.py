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

import tensorflow as tf
import logging
from tqdm import tqdm


def classification_eval(model: tf.keras.Model, data_loader: tf.data.Dataset, limit=None):
    """
    Evaluate a classification model

    Args:
        model: a model to evaluate
        data_loader: tensorflow evaluation dataset
        limit (int): optionally evaluate the model over less images than in the dataset

    Returns:
        model accuracy
        total number of images evaluated

    """
    logging.info(f'Start classification evaluation')
    acc = tf.keras.metrics.Accuracy()
    total = 0
    for data in tqdm(data_loader, desc="Classification evaluation"):
        images, labels = data
        # calculate outputs by running images through the network
        outputs = model(images)
        # the class with the highest energy is what we choose as prediction
        predicted = tf.argmax(outputs, 1)
        acc.update_state(labels, predicted)
        total += labels.shape[0]
        if limit and total >= int(limit):
            break

    logging.info(f'Num of images: {total}, Accuracy: {round(100 * acc.result().numpy(), 2)} %')

    return acc.result().numpy(), total


def get_representative_dataset(data_loader: tf.data.Dataset, n_iters: int, data_loader_key: int = 0, preprocess=None):
    """
    A function to generate a representative dataset generator. The class can iterate the rep. dataset over and
    over if there are less images in the dataset than required iterations.
    Args:
        data_loader (tf.data.Dataset): a tendorflow dataset
        n_iters (int): number of iterations
        data_loader_key (int): index of images in data_loader output (usually the output is a tuple: [image, label])
        preprocess (callable): a function to preprocess a batch of images

    Returns:
        A representative dataset generator

    """

    class RepresentativeDataset:
        def __init__(self, in_data_loader):
            self.dl = in_data_loader
            self.iter = iter(self.dl)

        def __call__(self):
            for _ in range(n_iters):
                try:
                    x = next(self.iter)[data_loader_key]
                except StopIteration:
                    self.iter = iter(self.dl)
                    x = next(self.iter)[data_loader_key]
                if preprocess is not None:
                    x = preprocess(x)
                yield [x.numpy()]

        def __len__(self):
            return n_iters

    return RepresentativeDataset(data_loader)

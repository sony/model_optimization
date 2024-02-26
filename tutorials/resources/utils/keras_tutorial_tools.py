import tensorflow as tf
import keras
import model_compression_toolkit as mct
import os
import torchvision

import os
import subprocess
import tarfile

def imgnet_download():
    imagenet_dir = 'imagenet'
    val_dir = os.path.join(imagenet_dir, 'val')
    
    if not os.path.isdir(imagenet_dir):
        os.makedirs(imagenet_dir, exist_ok=True)
        subprocess.run(['wget', 'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz', '-P', imagenet_dir])
        subprocess.run(['wget', 'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar', '-P', imagenet_dir])
        
        # Extract the downloaded files
        with tarfile.open(os.path.join(imagenet_dir, 'ILSVRC2012_devkit_t12.tar.gz'), 'r:gz') as tar:
            tar.extractall(path=imagenet_dir)
        with tarfile.open(os.path.join(imagenet_dir, 'ILSVRC2012_img_val.tar'), 'r:') as tar:
            tar.extractall(path=val_dir)
    
def imagenet_preprocess_input(images, labels):
    return tf.keras.applications.mobilenet_v2.preprocess_input(images), labels

from typing import Generator


def get_validation_dataset_fraction(fraction, TEST_DATASET_FOLDER, BATCH_SIZE) -> tf.data.Dataset:
    """Load a fraction of the validation dataset for evaluation.

    Args:
        fraction (float, optional): Fraction of the dataset to load. Defaults to 1.0 (i.e., the entire dataset).

    Returns:
        tf.data.Dataset: A fraction of the validation dataset.
    """
    assert 0 < fraction <= 1, "Fraction must be between 0 and 1."

    # Load the dataset to determine the total number of samples
    initial_dataset = tf.keras.utils.image_dataset_from_directory(
        directory=TEST_DATASET_FOLDER,
        batch_size=1,  # Use batch size of 1 to easily count samples
        image_size=[224, 224],
        shuffle=False,
        crop_to_aspect_ratio=True,
        interpolation='bilinear')

    total_samples = initial_dataset.cardinality().numpy()
    samples_to_take = int(total_samples * fraction)

    # Now, load the dataset again with the desired batch size and take the necessary number of samples
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory=TEST_DATASET_FOLDER,
        batch_size=BATCH_SIZE,
        image_size=[224, 224],
        shuffle=False,
        crop_to_aspect_ratio=True,
        interpolation='bilinear')

    # Preprocess the dataset
    dataset = dataset.map(lambda x, y: (imagenet_preprocess_input(x, y)))

    # Take the calculated number of samples (adjusted for batch size)
    dataset = dataset.take(samples_to_take // BATCH_SIZE + (1 if samples_to_take % BATCH_SIZE else 0))

    return dataset

from typing import Generator

def get_representative_dataset(fraction, REPRESENTATIVE_DATASET_FOLDER, BATCH_SIZE) -> Generator:
    """A function that loads a fraction of the dataset and returns a representative dataset generator.

    Args:
        fraction (float): The fraction of the dataset to load. Defaults to 1.0 (the entire dataset).

    Returns:
        Generator: A generator yielding batches of preprocessed images.
    """
    assert 0 < fraction <= 1, "Fraction must be between 0 and 1."

    print('Loading dataset, this may take a few minutes ...')    
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory=REPRESENTATIVE_DATASET_FOLDER,
        batch_size=BATCH_SIZE,
        image_size=[224, 224],
        shuffle=True,
        crop_to_aspect_ratio=True,
        interpolation='bilinear')

    # Preprocess the data
    dataset = dataset.map(lambda x, y: (imagenet_preprocess_input(x, y)))

    # Determine the total number of batches in the dataset
    total_batches = dataset.cardinality().numpy()
    if total_batches == tf.data.experimental.INFINITE_CARDINALITY:
        raise ValueError("Dataset size is infinite. A finite dataset is required to compute a fraction.")

    # Calculate the number of batches to use, based on the specified fraction
    batches_to_use = int(total_batches * fraction)

    def representative_dataset() -> Generator:
        """A generator function that yields batches of preprocessed images."""
        for image_batch, _ in dataset.take(batches_to_use):
            yield image_batch.numpy()

    print('images in representative dataset: '+ str(BATCH_SIZE*batches_to_use))

    return representative_dataset



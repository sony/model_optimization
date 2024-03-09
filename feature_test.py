
import argparse

from tensorflow.keras.applications.mobilenet import MobileNet

import model_compression_toolkit as mct
import tempfile

"""
This tutorial demonstrates how a model (more specifically, MobileNetV1) can be
quantized and optimized using the Model Compression Toolkit (MCT). 
"""

####################################
# Preprocessing images
####################################
import cv2
import numpy as np

MEAN = 127.5
STD = 127.5
RESIZE_SCALE = 256 / 224
SIZE = 224


def resize(x):
    resize_side = max(RESIZE_SCALE * SIZE / x.shape[0], RESIZE_SCALE * SIZE / x.shape[1])
    height_tag = int(np.round(resize_side * x.shape[0]))
    width_tag = int(np.round(resize_side * x.shape[1]))
    resized_img = cv2.resize(x, (width_tag, height_tag))
    offset_height = int((height_tag - SIZE) / 2)
    offset_width = int((width_tag - SIZE) / 2)
    cropped_img = resized_img[offset_height:offset_height + SIZE, offset_width:offset_width + SIZE]
    return cropped_img


def normalization(x):
    return (x - MEAN) / STD


def argument_handler():
    parser = argparse.ArgumentParser()
#    parser.add_argument('--representative_dataset_dir', type=str, required=True, default=None,
#                        help='folder path for the representative dataset.')
    parser.add_argument('--batch_size', type=int, default=5,
                        help='batch size for the representative data.')
    parser.add_argument('--num_calibration_iterations', type=int, default=5,
                        help='number of iterations for calibration.')
    return parser.parse_args()


if __name__ == '__main__':

    # Parse arguments
    args = argument_handler()

    # Set the batch size of the images at each calibration iteration.
    batch_size = args.batch_size

    # Set the path to the folder of images to load and use for the representative dataset.
    # Notice that the folder have to contain at least one image.
    from tensorflow.keras.datasets import cifar10

    # Load CIFAR-10 dataset
    (x_train, _), (test, _) = cifar10.load_data()

    # Preprocess the images in the dataset
    x_train_preprocessed = np.array([normalization(resize(image)) for image in test])

    # Create a representative data generator from the preprocessed dataset
    def image_data_generator(batch_size):
        for i in range(0, len(x_train_preprocessed), batch_size):
            yield x_train_preprocessed[i:i + batch_size]

    # Convert the generator to an iterator to use its 'next' method
    image_data_loader = iter(image_data_generator(batch_size))


    def representative_data_gen() -> list:
        for _ in range(args.num_calibration_iterations):
            yield next(image_data_loader)

    target_platform_cap = mct.get_target_platform_capabilities('tensorflow', 'default')

    model = MobileNet()

    quantized_model, quantization_info = mct.ptq.keras_post_training_quantization(model,
                                                                                  representative_data_gen,
                                                                                  target_platform_capabilities=target_platform_cap)

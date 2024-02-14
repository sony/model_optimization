# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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

"""
This tutorial demonstrates how the Model Compression Toolkit (MCT) prepares a model for Quantization Aware
Training. A model is trained on the MNIST dataset and then quantized and being QAT-ready by the MCT and
returned to the user. A QAT-ready model is a model with certain layers wrapped by a QuantizeWrapper with
the requested quantizers.
The user can now Fine-Tune the QAT-ready model. Finally, the model is finalized by the MCT which means the
MCT replaces the QuantizeWrappers with their native layers and quantized weights.
"""

import argparse

import numpy as np
import tensorflow as tf
from keras import Model, layers, datasets

from model_compression_toolkit import DefaultDict
from model_compression_toolkit.target_platform_capabilities.target_platform.op_quantization_config import AttributeQuantizationConfig
from model_compression_toolkit.constants import FLOAT_BITWIDTH
from model_compression_toolkit.target_platform_capabilities.constants import KERNEL_ATTR, KERAS_KERNEL, BIAS_ATTR, BIAS

import model_compression_toolkit as mct
import tempfile

tp = mct.target_platform


def get_tpc():
    """
    Assuming a target hardware that uses a power-of-2 threshold for activations and
    a symmetric threshold for the weights. The activations are quantized to 3 bits, and the weights
    are quantized to 3 bits. Our assumed hardware does not require quantization of some layers
    (e.g. Flatten & Droupout).
    This function generates a TargetPlatformCapabilities with the above specification.

    Returns:
         TargetPlatformCapabilities object
    """

    # define a default quantization config for all non-specified weights attributes.
    default_weight_attr_config = AttributeQuantizationConfig(
        weights_quantization_method=tp.QuantizationMethod.POWER_OF_TWO,
        weights_n_bits=8,
        weights_per_channel_threshold=False,
        enable_weights_quantization=False,
        lut_values_bitwidth=None)

    # define a quantization config to quantize the kernel (for layers where there is a kernel attribute).
    kernel_base_config = AttributeQuantizationConfig(
        weights_quantization_method=tp.QuantizationMethod.SYMMETRIC,
        weights_n_bits=2,
        weights_per_channel_threshold=True,
        enable_weights_quantization=True,
        lut_values_bitwidth=None)

    # define a quantization config to quantize the bias (for layers where there is a bias attribute).
    bias_config = AttributeQuantizationConfig(
        weights_quantization_method=tp.QuantizationMethod.POWER_OF_TWO,
        weights_n_bits=FLOAT_BITWIDTH,
        weights_per_channel_threshold=False,
        enable_weights_quantization=False,
        lut_values_bitwidth=None)

    # Create a default OpQuantizationConfig where we use default_weight_attr_config as the default
    # AttributeQuantizationConfig for weights with no specific AttributeQuantizationConfig.
    # MCT will compress a layer's kernel and bias according to the configurations that are
    # set in KERNEL_ATTR and BIAS_ATTR that are passed in attr_weights_configs_mapping.
    default_config = tp.OpQuantizationConfig(
        default_weight_attr_config=default_weight_attr_config,
        attr_weights_configs_mapping={KERNEL_ATTR: kernel_base_config,
                                      BIAS_ATTR: bias_config},
        activation_quantization_method=tp.QuantizationMethod.POWER_OF_TWO,
        activation_n_bits=3,
        enable_activation_quantization=True,
        quantization_preserving=False,
        fixed_scale=None,
        fixed_zero_point=None,
        simd_size=None)

    # Set default QuantizationConfigOptions in new TargetPlatformModel to be used when no other
    # QuantizationConfigOptions is set for an OperatorsSet.
    default_configuration_options = tp.QuantizationConfigOptions([default_config])
    tp_model = tp.TargetPlatformModel(default_configuration_options)
    with tp_model:
        default_qco = tp.get_default_quantization_config_options()
        # Group of OperatorsSets that should not be quantized.
        tp.OperatorsSet("NoQuantization",
                        default_qco.clone_and_edit(enable_activation_quantization=False)
                        .clone_and_edit_weight_attribute(enable_weights_quantization=False))
        # Group of linear OperatorsSets such as convolution and matmul.
        tp.OperatorsSet("LinearOp")

    tpc = tp.TargetPlatformCapabilities(tp_model)
    with tpc:
        # No need to quantize Flatten and Dropout layers
        tp.OperationsSetToLayers("NoQuantization", [layers.Flatten, layers.Dropout])
        # Assign the framework layers' attributes to KERNEL_ATTR and BIAS_ATTR that were used during creation
        # of the default OpQuantizationConfig.
        tp.OperationsSetToLayers("LinearOp", [layers.Dense, layers.Conv2D],
                                 attr_mapping={KERNEL_ATTR: DefaultDict(default_value=KERAS_KERNEL),
                                               BIAS_ATTR: DefaultDict(default_value=BIAS)})
    return tpc


def get_model(_num_classes, _input_shape):
    """
    Generate example keras model
    Args:
        _num_classes: Number of classes (10 for MNIST)
        _input_shape: input image shape (28x28x1 for MNIST)

    Returns:
        Keras model

    """
    _input = layers.Input(shape=_input_shape)
    x = layers.Conv2D(16, 3, strides=2, padding='same', activation='relu')(_input)
    x = layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(_num_classes, activation='softmax')(x)
    return Model(inputs=_input, outputs=x)


def get_dataset(_num_classes):
    """
    This function returns the MNIST dataset

    Args:
        _num_classes: Number of classes (10 for MNIST)

    Returns:
        x_train: A tuple of numpy array of training images
        y_train: A tuple of numpy array of training labels
        x_test: A tuple of numpy array of test images
        y_test: A tuple of numpy array of test labels
    """

    # Load the data and split it between train and test sets
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

    # Normalize images
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # Add Channels axis to data
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(y_train, _num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, _num_classes)
    return x_train, y_train, x_test, y_test


def gen_representative_dataset(_images, num_calibration_iterations):
    # Return a Callable representative dataset for calibration purposes.
    # The function should be called without any arguments, and should return a list numpy arrays (array
    # for each model's input).
    # In this tutorial, each time the representative dataset is called it returns a list containing a single
    # MNIST image of shape (1, 28, 28, 1).
    def _generator():
        for _ind in range(num_calibration_iterations):
            yield [_images[_ind][np.newaxis, ...]]
    return _generator


def argument_handler():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size for model training.')
    parser.add_argument('--num_epochs', type=int, default=15,
                        help='number of epochs for model training.')
    parser.add_argument('--num_calibration_iterations', type=int, default=10,
                        help='number of iterations for calibration - model quantization before fine-tuning.')
    return parser.parse_args()


if __name__ == "__main__":
    """
    The code below is an example code of a user for fine tuning a float model with the MCT Quantization
    Aware Training API. 
    """

    # Parse arguments
    args = argument_handler()

    # init parameters
    num_classes = 10
    input_shape = (28, 28, 1)
    batch_size = args.batch_size
    epochs = args.num_epochs

    # init model
    model = get_model(num_classes, input_shape)
    model.summary()

    # init dataset
    x_train, y_train, x_test, y_test = get_dataset(num_classes)

    # train float model
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

    # evaluate float model
    score = model.evaluate(x_test, y_test, verbose=0)
    print(f"Float model test accuracy: {score[1]:02.4f}")

    # prepare a representative dataset callable from the MNIST training images for calibrating the initial
    # quantization parameters by the MCT.
    representative_dataset = gen_representative_dataset(x_train, args.num_calibration_iterations)

    # prepare model for QAT with MCT and return to user for fine-tuning. Due to the relatively easy
    # task of quantizing model trained on MNIST, a custom TPC is used in this example to demonstrate
    # the degradation caused by post training quantization.
    qat_model, _, _ = mct.qat.keras_quantization_aware_training_init(model,
                                                                     representative_dataset,
                                                                     core_config=mct.core.CoreConfig(),
                                                                     target_platform_capabilities=get_tpc())

    # Evaluate QAT-ready model accuracy from MCT. This model is fully quantized with QuantizeWrappers
    # for weights and tf.quantization.fake_quant_with_min_max_vars for activations
    qat_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"], run_eagerly=True)
    score = qat_model.evaluate(x_test, y_test, verbose=0)
    print(f"PTQ model test accuracy: {score[1]:02.4f}")

    # fine-tune QAT model from MCT to recover the lost accuracy.
    qat_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

    # Evaluate accuracy after fine-tuning.
    score = qat_model.evaluate(x_test, y_test, verbose=0)
    print(f"QAT model test accuracy: {score[1]:02.4f}")

    # Finalize QAT model: remove QuantizeWrappers and keep weights quantized as fake-quant values
    quantized_model = mct.qat.keras_quantization_aware_training_finalize(qat_model)

    # Re-evaluate accuracy after finalizing the model (should have the same accuracy as QAT model
    # after fine-tuning. Accuracy should be the same as before calling the finalize function.
    quantized_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    score = quantized_model.evaluate(x_test, y_test, verbose=0)
    print(f"Quantized model test accuracy: {score[1]:02.4f}")

    # Export quantized model to Keras.
    # For more details please see: https://github.com/sony/model_optimization/blob/main/model_compression_toolkit
    # /exporter/README.md
    _, keras_file_path = tempfile.mkstemp('.keras')  # Path of exported model
    mct.exporter.keras_export_model(model=quantized_model,
                                    save_model_path=keras_file_path)

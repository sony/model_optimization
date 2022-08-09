
import tensorflow as tf
from keras.datasets import mnist
from keras import Model, layers, datasets
import model_compression_toolkit as mct
import numpy as np


def get_tpc():
    # Generate a TargetPlatformModel with power of two quantization with 3 bits for
    # activations and 2 bits for weights
    tp = mct.target_platform
    default_config = tp.OpQuantizationConfig(
        activation_quantization_method=tp.QuantizationMethod.POWER_OF_TWO,
        weights_quantization_method=tp.QuantizationMethod.POWER_OF_TWO,
        activation_n_bits=3,
        weights_n_bits=2,
        weights_per_channel_threshold=True,
        enable_weights_quantization=True,
        enable_activation_quantization=True,
        quantization_preserving=False,
        fixed_scale=1.0,
        fixed_zero_point=0,
        weights_multiplier_nbits=0)

    default_configuration_options = tp.QuantizationConfigOptions([default_config])
    tp_model = tp.TargetPlatformModel(default_configuration_options)
    with tp_model:
        tp.OperatorsSet("NoQuantization",
                        tp.get_default_quantization_config_options().clone_and_edit(
                            enable_weights_quantization=False,
                            enable_activation_quantization=False))

    tpc = tp.TargetPlatformCapabilities(tp_model)
    with tpc:
        # No need to quantize Flatten and Dropout layers
        tp.OperationsSetToLayers("NoQuantization", [layers.Flatten,
                                                    layers.Dropout])

    return tpc


def get_model(_num_classes, _input_shape):
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
    return (x_train, y_train), (x_test, y_test)


def gen_representative_dataset(_images):
    def _generator():
        for _img in _images:
            yield [_img[np.newaxis, ...]]
    return _generator().__next__


if __name__ == "__main__":
    # init parameters
    num_classes = 10
    input_shape = (28, 28, 1)
    batch_size = 128
    epochs = 15

    # init model
    model = get_model(num_classes, input_shape)
    model.summary()

    # init dataset
    (x_train, y_train), (x_test, y_test) = get_dataset(num_classes)

    # train float model
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

    # evaluate float model
    score = model.evaluate(x_test, y_test, verbose=0)
    print(f"Float model test accuracy: {score[1]:02.4f}")

    # prepare model for QAT with MCT and return to user for fine-tuning
    representative_dataset = gen_representative_dataset(x_train)

    qat_model, _, _ = mct.keras_quantization_aware_training_init(model,
                                                                 representative_dataset,
                                                                 core_config=mct.CoreConfig(n_iter=10),
                                                                 target_platform_capabilities=get_tpc())

    # Evaluate QAT-ready model accuracy from MCT. This model is fully quantized with QuantizeWrappers
    # for weights and tf.quantization.fake_quant_with_min_max_vars for activations
    qat_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    score = qat_model.evaluate(x_test, y_test, verbose=0)
    print(f"PTQ model test accuracy: {score[1]:02.4f}")

    # fine-tune QAT model from MCT
    qat_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

    # Evaluate accuracy after fine-tuning
    score = qat_model.evaluate(x_test, y_test, verbose=0)
    print(f"QAT model test accuracy: {score[1]:02.4f}")

    # Finalize QAT model: remove QuantizeWrappers and keep weights quantized as fake-quant values
    quantized_model = mct.keras_quantization_aware_training_finalize(qat_model)

    # Re-evaluate accuracy after finalizing the model (should have the same accuracy as QAT model
    # after fine-tuning
    quantized_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    score = quantized_model.evaluate(x_test, y_test, verbose=0)
    print(f"Quantized model test accuracy: {score[1]:02.4f}")


import tensorflow as tf
from keras.datasets import mnist
from keras import Model, layers, datasets
import model_compression_toolkit as mct
from model_compression_toolkit.qat.keras.quantization_facade import DEFAULT_KERAS_TPC as default_tpc
import numpy as np


# init model
num_classes = 10
input_shape = (28, 28, 1)

_input = layers.Input(shape=input_shape)
x = layers.Conv2D(16, 3, strides=2, padding='same', activation='relu')(_input)
x = layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')(x)
x = layers.Flatten()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(num_classes, activation='softmax')(x)
model = Model(inputs=_input, outputs=x)

model.summary()

# init dataset

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# Normalize images
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Add Channels axis to data
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# train float model
batch_size = 128
epochs = 15

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

# evaluate float model
score = model.evaluate(x_test, y_test, verbose=0)
print(f"Float model test accuracy: {score[1]:02.4f}")


def gen_representative_dataset():
    def _generator():
        for _img in x_train:
            yield [_img[np.newaxis, ...]]
    return _generator().__next__


representative_dataset = gen_representative_dataset()
default_tpc.tp_model.default_qco.base_config.weights_n_bits = 2
default_tpc.tp_model.default_qco.base_config.activation_n_bits = 3

qat_model, _, custom_objects = mct.keras_quantization_aware_training_init(model,
                                                                          representative_dataset,
                                                                          core_config=mct.CoreConfig(n_iter=10),
                                                                          target_platform_capabilities=default_tpc)

qat_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
score = qat_model.evaluate(x_test, y_test, verbose=0)
print(f"PTQ model test accuracy: {score[1]:02.4f}")

qat_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

score = qat_model.evaluate(x_test, y_test, verbose=0)
print(f"QAT model test accuracy: {score[1]:02.4f}")

quantized_model = mct.keras_quantization_aware_training_finalize(qat_model)

quantized_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
score = quantized_model.evaluate(x_test, y_test, verbose=0)
print(f"Quantized model test accuracy: {score[1]:02.4f}")
quantized_model.save('/tmp/qat_example_quantized_model.h5')

from inspect import signature


class A:
    def __init__(self):
        self.kwargs = {}
        _signature = self.get_sig()
        for k, v in _signature.parameters.items():
            assert v.default is v.empty, f"Parameter {k} doesn't have a default value"

    @classmethod
    def get_sig(cls):
        return signature(cls)

    @property
    def get_kwargs(self):
        return self.kwargs


class B(A):
    def __init__(self, aa, a=1):
        super().__init__()

b=B(5)




import numpy as np
import tensorflow as tf
import keras

import tensorflow_model_optimization as tfmot

import tempfile


input_shape = [20]
x_train = np.random.randn(1, 20).astype(np.float32)
y_train = tf.keras.utils.to_categorical(np.random.randn(1), num_classes=20)

_in = tf.keras.layers.Input(shape=input_shape)
_out = tf.keras.layers.Dense(20)(_in)
model = tf.keras.Model(inputs=_in, outputs=_out)

quant_aware_model = tfmot.quantization.keras.quantize_model(model)
quant_aware_model.summary()













m=keras.models.Sequential([tf.keras.layers.Conv2D(5,3),
                           tf.keras.layers.Lambda(lambda x: tf.quantization.fake_quant_with_min_max_args(x, min=-4, max=8, num_bits=4))])

if True:
    model_input = tf.keras.layers.Input(shape=(5, 5, 3))
    a = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=1, padding='same')(model_input)
    a = tf.equal(a, model_input)
    # a = tf.cast(a, tf.float32)
    # b = model_input * a
    b = tf.keras.layers.Multiply()([model_input, a])
    model = tf.keras.Model(inputs=model_input, outputs=b)

if False:
    mobnetv2_no_bn = keras.models.load_model('/tmp/qat/mobnetv2_no_bn.h5')
    mobnetv2_with_bn = keras.models.load_model('/tmp/qat/mobnetv2_with_bn.h5')
    _input = np.random.normal(size=(1, 32, 32, 3)).astype(np.float32)
    out_no_bn = mobnetv2_no_bn(_input)[1].numpy()
    out_with_bn = mobnetv2_with_bn(_input)[1].numpy()


if True:  # check BN identity
    # _input = np.array(np.arange(4).reshape((2, 2)).astype(np.float32))
    #
    model_input = tf.keras.layers.Input(shape=(2, 2))
    bn = tf.keras.layers.BatchNormalization(momentum=0.99, trainable=True, center=False, scale=False)
    bn2 = tf.keras.layers.BatchNormalization(epsilon=0.0, trainable=True, center=True, scale=True)
    out = bn2(bn(model_input))
    model = tf.keras.Model(inputs=model_input, outputs=out)

    # Instantiate an optimizer.
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    # Instantiate a loss function.
    loss_fn = keras.losses.MeanSquaredError()

    # Prepare the training dataset.
    batch_size = 10
    x_train, y_train = 2*np.random.randn(200, 2, 2)+1, 2*np.random.randn(200, 2, 2)+1
    x_val, y_val = 2*np.random.randn(200, 2, 2)+1, 2*np.random.randn(200, 2, 2)+1

    # Prepare the training dataset.
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    # Prepare the validation dataset.
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(batch_size)

    model.compile(
        optimizer=optimizer, loss=loss_fn,
        metrics=[keras.metrics.Accuracy()],
    )

    print("Fit model on training data")
    history = model.fit(
        x_train,
        y_train,
        epochs=2,
        validation_data=(x_val, y_val),
    )

if True:  # check BN identity
    _input = np.array(np.arange(4).reshape((2, 2)).astype(np.float32))

    model_input = tf.keras.layers.Input(shape=(2, 2))
    bn = tf.keras.layers.BatchNormalization(momentum=0.99, trainable=True, center=True, scale=True)
    out = bn(model_input)
    model = tf.keras.Model(inputs=model_input, outputs=out)

    a = 0.6
    b = 0.4
    # bn.gamma.assign(np.sqrt(a+bn.epsilon)*np.ones(2).astype(np.float32))
    # bn.beta.assign(b*np.ones(2).astype(np.float32))
    bn.moving_variance.assign(a*np.ones(2).astype(np.float32))
    bn.moving_mean.assign(b*np.ones(2).astype(np.float32))

    print(f'init: moving_mean={bn.moving_mean.numpy()}, moving_variance={bn.moving_variance.numpy()}')
    for _ in range(100):
        # a = model.call(tf.convert_to_tensor(_input[np.newaxis, ...]), training=True)
        a = model(_input[np.newaxis, ...], training=True)
        print(f'{_}: moving_mean={bn.moving_mean.numpy()}, moving_variance={bn.moving_variance.numpy()}')

if False:
    num_keypoints = 3
    N = 4
    top_k_points = 2
    tag_length = 1
    tags_shape = (1, num_keypoints, N) if tag_length == 1 else (1, num_keypoints, N, tag_length)
    tags = np.arange(num_keypoints*N*tag_length).reshape(tags_shape)
    ind = np.array([[0, 2], [1, 3], [0, 0]]).reshape((1, num_keypoints, top_k_points))
    tag_k = tf.gather(tags, ind, axis=2, batch_dims=2).numpy()

    print('params shape:', tags.shape)
    print('indices shape:', ind.shape)
    print('output shape:', tag_k.shape)

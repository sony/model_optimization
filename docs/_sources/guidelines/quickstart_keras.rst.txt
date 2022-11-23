:orphan:

.. _ug-quickstart-keras:

=================================================
MCT Quickstart Guideline for Keras models
=================================================

Here is an example of a code that shows how to use MCT with Keras models.


Import MCT and MobileNetV1 from Keras applications:

.. code-block:: python

    import model_compression_toolkit as mct
    from tensorflow.keras.applications.mobilenet import MobileNet

|

Data preprocessing functions:

.. code-block:: python

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


|

Initialize data loader:

.. code-block:: python

    # Set the batch size of the images at each calibration iteration.
    batch_size = 50

    # Set the path to the folder of images to load and use for the representative dataset.
    # Notice that the folder have to contain at least one image.
    folder = '/path/to/images/folder'

    # Create a representative data generator, which returns a list of images.
    # The images can be preprocessed using a list of preprocessing functions.
    from model_compression_toolkit import FolderImageLoader
    image_data_loader = FolderImageLoader(folder,
                                          preprocessing=[resize, normalization],
                                          batch_size=batch_size)

    # Create a Callable representative dataset for calibration purposes.
    # The function should be called without any arguments, and should return a list numpy arrays (array for each
    # model's input).
    # For example: A model has two input tensors - one with input shape of [32 X 32 X 3] and the second with
    # an input shape of [224 X 224 X 3]. We calibrate the model using batches of 20 images.
    # Calling representative_data_gen() should return a list
    # of two numpy.ndarray objects where the arrays' shapes are [(20, 3, 32, 32), (20, 3, 224, 224)].
    def representative_data_gen() -> list:
        return [image_data_loader.sample()]

|

Get a TargetPlatformCapabilities:

.. code-block:: python

    # Get a TargetPlatformModel object that models the hardware for the quantized model inference.
    # The model determines the quantization methods to use during the MCT optimization process.
    # Here, for example, we use the default target platform model that is attached to a Tensorflow
    # layers representation.
    target_platform_cap = mct.get_target_platform_capabilities('tensorflow', 'default')

|

Run Post Training Quantization:

.. code-block:: python

    # Create a model and quantize it using the representative_data_gen as the calibration images.
    # Set the number of calibration iterations to 10.
    model = MobileNet()

    quantized_model, quantization_info = mct.keras_post_training_quantization(model,
                                                                              representative_data_gen,
                                                                              target_platform_capabilities=target_platform_cap,
                                                                              n_iter=10)

|


import argparse

#from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.inception_v3 import InceptionV3
import model_compression_toolkit as mct
import tempfile
import tensorflow_hub as hub
from tensorflow.keras.applications.densenet import DenseNet121
"""
This tutorial demonstrates how a model (more specifically, MobileNetV1) can be
quantized and optimized using the Model Compression Toolkit (MCT). 
"""

####################################
# Preprocessing images
####################################
import cv2
import numpy as np

SIZE = 224  # Target size

def resize(image):
    """
    Resize the image to the target size while maintaining aspect ratio.
    """
    resize_side = max(SIZE / image.shape[0], SIZE / image.shape[1])
    height_tag = int(np.round(resize_side * image.shape[0]))
    width_tag = int(np.round(resize_side * image.shape[1]))
    resized_img = cv2.resize(image, (width_tag, height_tag))
    offset_height = int((height_tag - SIZE) / 2)
    offset_width = int((width_tag - SIZE) / 2)
    cropped_img = resized_img[offset_height:offset_height + SIZE, offset_width:offset_width + SIZE]
    return cropped_img

def normalization(image):
    """
    Normalize the image by scaling pixel values to [0, 1].
    """
    return image / 255.0

def preprocess_images(images):
    """
    Apply resizing and normalization to a batch of images.
    """
    preprocessed_images = np.array([normalization(resize(image)) for image in images])
    return preprocessed_images

def argument_handler():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for the representative data.')
    parser.add_argument('--num_calibration_iterations', type=int, default=1, help='number of iterations for calibration.')
    return parser.parse_args()

if __name__ == '__main__':
    args = argument_handler()
    from tensorflow.keras.datasets import cifar10
    # Load CIFAR-10 dataset
    (_, _), (test_images, _) = cifar10.load_data()

    # Preprocess the images in the dataset
    test_images_preprocessed = preprocess_images(test_images)

    # Further processing and model loading here
    model = DenseNet121()
    # Set the batch size of the images at each calibration iteration.
    batch_size = args.batch_size

    # Set the path to the folder of images to load and use for the representative dataset.
    # Notice that the folder have to contain at least one image.
    

    # Load CIFAR-10 dataset

    # Preprocess the images in the dataset
    #x_train_preprocessed = np.array([normalization(resize(image)) for image in test])

    # Create a representative data generator from the preprocessed dataset
    def image_data_generator(batch_size):
        for i in range(0, len(test_images_preprocessed), batch_size):
            yield test_images_preprocessed[i:i + batch_size]

    # Convert the generator to an iterator to use its 'next' method
    image_data_loader = iter(image_data_generator(batch_size))


    def representative_data_gen() -> list:
        for _ in range(args.num_calibration_iterations):
            yield next(image_data_loader)

    target_platform_cap = mct.get_target_platform_capabilities('tensorflow', 'default')


    model = InceptionV3()
    model.save('incepmodelpreq.keras')

    # from tensorflow.keras.applications.inception_v3 import InceptionV3
    from tensorflow.keras.applications.densenet import DenseNet121
    model = DenseNet121()

    #model_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
    #model = hub.load(model_handle)
    '''
    quantized_model, quantization_info = mct.ptq.keras_post_training_quantization(model,
                                                                                  representative_data_gen,
                                                                                  target_platform_capabilities=target_platform_cap)
    
    quantized_model.save('model.keras')
    '''
    # Enable Mixed-Precision config. For the sake of running faster, the hessian-based scores are disabled in this tutorial
    mp_config = mct.core.MixedPrecisionQuantizationConfig(
        num_of_images=32,
        use_hessian_based_scores=True)
    core_config = mct.core.CoreConfig(mixed_precision_config=mp_config)
    # Specify the target platform capability (TPC)
    tpc = mct.get_target_platform_capabilities("tensorflow", 'imx500', target_platform_version='v1')
    tpc = mct.get_target_platform_capabilities('tensorflow', 'default')


    # Get Resource Utilization information to constraint your model's memory size. Retrieve a ResourceUtilization object with helpful information of each resource metric, to constraint the quantized model to the desired memory size.
    resource_utilization_data = mct.core.keras_resource_utilization_data(model,
                                    representative_data_gen,
                                    core_config=core_config,
                                    target_platform_capabilities=tpc)

    # Set a constraint for each of the Resource Utilization metrics.
    # Create a ResourceUtilization object to limit our returned model's size. Note that this values affects only layers and attributes
    # that should be quantized (for example, the kernel of Conv2D in Keras will be affected by this value,
    # while the bias will not)
    # examples:
    weights_compression_ratio = 0.75  # About 0.75 of the model's weights memory size when quantized with 8 bits.
    resource_utilization = mct.core.ResourceUtilization(resource_utilization_data.weights_memory * weights_compression_ratio)

    quantized_model, quantization_info = mct.ptq.keras_post_training_quantization(
        model,
        representative_data_gen,
        target_resource_utilization=resource_utilization,
        core_config=core_config,
        target_platform_capabilities=tpc)







    # Print all the layers of the model
    for i, layer in enumerate(model.layers):
        print(f"Layer {i}: {layer.name}")

    print('here')

    for i in range(0,200):
        try:
            optimal_threshold = quantized_model.layers[i].activation_holder_quantizer.get_config()['threshold'][0]
        except Exception:
            optimal_threshold = 0
        print(f"Layer {i}: {optimal_threshold}")

    print('done')
   

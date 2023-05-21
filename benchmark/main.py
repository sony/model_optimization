import argparse

from benchmark.quant import quant
from benchmark.sources.sources import get_model_source


def argument_handler():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', '-m', type=str, required=True,
                        help='The name of the model to run')
    parser.add_argument('--model_source', type=str, default='torchvision',
                        help='The source of the model out of supported packages',
                        choices=['torchvision', 'timm', 'ultralytics'])
    parser.add_argument('--dataset_name', type=str, default='IMAGENET',
                        help='The name of the dataset used for the pre-trained model')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for model evaluation')
    parser.add_argument('--validation_dataset_folder', type=str, default='',
                        help='Path to the validation dataset')
    parser.add_argument('--representative_dataset_folder', type=str, default='',
                        help='Path to the representative dataset used for quantization')
    parser.add_argument('--n_images', type=int, default=1024,
                        help='Number of images for representative dataset')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Image size required by the pretrained model')
    parser.add_argument('--random_seed', type=int, default=0,
                        help='Random seed')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    #################################################
    # Set arguments and parameters
    #################################################
    args = argument_handler()

    #################################################
    # Import the relevant source and pre-trained model
    #################################################
    ModelSrc = get_model_source(args)

    #################################################
    # Evaluate float model
    #################################################
    float_results = ModelSrc.evaluation(ModelSrc.get_model(), args)

    #################################################
    # Run model compression toolkit
    #################################################
    quantized_model = ModelSrc.quantize(args)

    #################################################
    # Evaluate quantized model
    #################################################
    quant_results = ModelSrc.evaluation(quantized_model, args)

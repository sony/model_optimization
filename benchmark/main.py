import argparse

from benchmark.quant import quant
from benchmark.sources.sources import get_model_source


def argument_handler():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', '-m', type=str, required=True,
                        help='The name of the model to run')
    parser.add_argument('--model_source', type=str, default='torchvision')
    parser.add_argument('--dataset_name', type=str, default='IMAGENET')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train_data_path', type=str, default='')
    parser.add_argument('--val_data_path', type=str, default='')
    parser.add_argument('--n_images', type=int, default=1024)

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
    model = ModelSrc.get_model()

    #################################################
    # Evaluate float model
    #################################################
    float_results = ModelSrc.evaluation(model, args)

    #################################################
    # Run model compression toolkit
    #################################################
    quantized_model = quant(ModelSrc, args)

    #################################################
    # Evaluate quantized model
    #################################################
    quant_results = ModelSrc.evaluation(quantized_model, args)







import argparse
import importlib

from benchmark.common.helpers import read_benchmark_list, write_benchmark_list
from benchmark.common.sources import find_modules


def argument_handler():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', '-m', type=str, required=True,
                        help='The name of the model to run')
    parser.add_argument('--model_library', type=str, default='torchvision',
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
    parser.add_argument('--benchmark_csv_list', type=str, default=None,
                        help='Runs benchmark test according the a list of models and parameters taken from a csv file')
    parser.add_argument('--validation_set_limit', type=int, default=None,
                        help='Limits the number of images taken for evaluation')


    args = parser.parse_args()
    return args


def quantization_flow(args):

    #################################################
    # Import the relevant models library and pre-trained model
    #################################################

    # Find relevant modules to import according to the model_library
    model_lib_module, quant_module = find_modules(args['model_library'])
    model_lib = importlib.import_module(model_lib_module)
    quant = importlib.import_module(quant_module)

    # Create ModelLibrary object and get the pre-trained model
    ml = model_lib.ModelLib(args)
    float_model = ml.get_model()

    #################################################
    # Evaluate float model
    #################################################
    float_results = ml.evaluate(float_model)

    #################################################
    # Run model compression toolkit
    #################################################
    target_platform_cap = quant.get_tpc()
    quantized_model, quantization_info = quant.quantize(float_model, ml.get_representative_dataset, target_platform_cap, args)

    #################################################
    # Evaluate quantized model
    #################################################
    quant_results = ml.evaluate(quantized_model)

    return float_results, quant_results, quantization_info


if __name__ == '__main__':

    #################################################
    # Set arguments and parameters
    #################################################
    args = argument_handler()

    if args.benchmark_csv_list is None:
        params = dict(args._get_kwargs())
        float_acc, quant_acc, quant_info = quantization_flow(params)
    else:
        models_list = read_benchmark_list(args.benchmark_csv_list)
        result_list = []
        params = dict(args._get_kwargs())
        for p in models_list:
            params.update(p)
            print(f"Testing model: {params['model_name']} from library: {params['model_library']}")
            res = {}
            res['model_name'] = params['model_name']
            res['model_library'] = params['model_library']
            res['dataset_name'] = params['dataset_name']
            res['float_acc'], res['quant_acc'], quant_info = quantization_flow(params)
            res['model_size'] = quant_info.final_kpi.weights_memory
            result_list.append(res)
        write_benchmark_list("model_quantization_results.csv", result_list, res.keys())


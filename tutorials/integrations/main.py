import argparse
import importlib
import logging
from integrations.common.helpers import write_results, read_models_list, parse_results
from integrations.common.helpers import find_modules
from integrations.common.consts import MODEL_NAME, MODEL_LIBRARY, OUTPUT_RESULTS_FILE, TARGET_PLATFORM_NAME, \
    TARGET_PLATFORM_VERSION


def argument_handler():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', '-m', type=str, required=False,
                        help='The name of the pre-trained model to run')
    parser.add_argument('--model_library', type=str, required=False,
                        help='The library that contains the pre-trained model')
    parser.add_argument('--validation_dataset_folder', type=str, required=False,
                        help='Path to the validation dataset')
    parser.add_argument('--representative_dataset_folder', type=str, required=False,
                        help='Path to the representative dataset used for quantization')
    parser.add_argument('--num_representative_images', type=int, default=1024,
                        help='Number of images for representative dataset')
    parser.add_argument('--dataset_name', type=str, default='',
                        help='Datset name for comment')
    parser.add_argument('--target_platform_name', type=str, default='default',
                        help='Specifies a target platform capabilites (tpc) name')
    parser.add_argument('--target_platform_version', type=str, default=None,
                        help='Specifies a target platform capabilites (tpc) version')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for accuracy evaluation')
    parser.add_argument('--random_seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--models_list_csv', type=str, default=None,
                        help='Run according to a list of models and parameters taken from a csv file')
    parser.add_argument('--validation_set_limit', type=int, default=None,
                        help='Limits the number of images taken for evaluation')

    args = parser.parse_args()
    return args


def quantization_flow(config):

    #################################################
    # Import the relevant models library and pre-trained model
    #################################################

    # Find relevant modules to import according to the model_library
    model_lib_module, quant_module = find_modules(config[MODEL_LIBRARY])
    model_lib = importlib.import_module(model_lib_module)
    quant = importlib.import_module(quant_module)

    # Create ModelLibrary object and get the pre-trained model
    ml = model_lib.ModelLib(config)
    float_model = ml.get_model()

    #################################################
    # Evaluate float model
    #################################################
    float_results = ml.evaluate(float_model)

    #################################################
    # Run model compression toolkit
    #################################################
    target_platform_cap = quant.get_tpc(config[TARGET_PLATFORM_NAME], config[TARGET_PLATFORM_VERSION])
    quantized_model, quantization_info = quant.quantize(float_model,
                                                        ml.get_representative_dataset,
                                                        target_platform_cap,
                                                        config)

    #################################################
    # Evaluate quantized model
    #################################################
    quant_results = ml.evaluate(quantized_model)

    return float_results, quant_results, quantization_info


if __name__ == '__main__':

    # Set arguments and parameters
    args = argument_handler()

    # Set logger level
    logging.getLogger().setLevel(logging.INFO)

    if args.models_list_csv is None:
        config = dict(args._get_kwargs())
        float_acc, quant_acc, quant_info = quantization_flow(config)
    else:
        models_list = read_models_list(args.models_list_csv)
        results_table = []
        config = dict(args._get_kwargs())
        for p in models_list:

            # Get next model and parameters from the list
            logging.info(f"Testing model: {p[MODEL_NAME]} from library: {p[MODEL_LIBRARY]}")
            config.update(p)

            # Run quantization flow and add results to the table
            float_acc, quant_acc, quant_info = quantization_flow(config)

            # Add results to the table
            res = parse_results(config, float_acc, quant_acc, quant_info)
            results_table.append(res)

        # Store results table
        write_results(OUTPUT_RESULTS_FILE, results_table, res.keys())


import csv
from os import path
import numpy as np
import importlib


def find_modules(lib):
    # pytorch
    if importlib.util.find_spec('integrations.pytorch.' + lib) is not None:
        model_lib_module = 'integrations.pytorch.' + lib + '.model_lib_' + lib
        quant_module = 'integrations.pytorch.quant'
    elif importlib.util.find_spec('integrations.keras.' + lib) is not None:
        model_lib_module = 'integrations.keras.' + lib + '.model_lib_' + lib
        quant_module = 'integrations.keras.quant'
    else:
        raise Exception(f'Error: model library {lib} was not found')
    return model_lib_module, quant_module


def read_models_list(filename):
    return csv.DictReader(open(path.join(filename)))


def write_results(filename, models_list, fieldnames):
    writer = csv.DictWriter(open(filename, 'w'), fieldnames=fieldnames)
    writer.writeheader()
    for item in models_list:
        writer.writerow(item)


def parse_results(params, float_acc, quant_acc, quant_info):
    res = {}
    res['ModelName'] = params['model_name']
    res['ModelLibrary'] = params['model_library']
    res['DatasetName'] = params['dataset_name']
    res['FloatAcc'] = round(float_acc, 4)
    res['QuantAcc'] = round(quant_acc, 4)
    res['QuantModelSize[MB]'] = round(quant_info.final_kpi.weights_memory / 1e6, 2)
    res['CompressionFactor'] = '4'
    res['QuantTechnique'] = 'PTQ'
    res['BitConfiguration'] = 'W8A8'

    return res

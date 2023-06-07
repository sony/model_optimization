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
    writer = csv.DictWriter(open(path.join("results",filename), 'w'), fieldnames=fieldnames)
    writer.writeheader()
    for item in models_list:
        item['float_acc'], item['quant_acc'] = np.round((item['float_acc'], item['quant_acc']),4)
        writer.writerow(item)


def parse_results(params, float_acc, quant_acc, quant_info):
    res = {}
    res['model_name'] = params['model_name']
    res['model_library'] = params['model_library']
    res['dataset_name'] = params['dataset_name']
    res['float_acc'] = float_acc
    res['quant_acc'] = quant_acc
    res['model_size'] = quant_info.final_kpi.weights_memory

    return res

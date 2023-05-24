import importlib

LIB_FW_DICT = {
    'torchvision': 'pytorch',
    'timm': 'pytorch',
    'ultrlytics': 'pytorch'
}
# def get_model_interface(args):
#     # todo: add torchvision object detection notion (for ssd)
#     if args.model_source == 'torchvision':
#         from .torchvision.model_torchvision import ModelTorchvision as ModelInterface
#     elif args.model_source == 'timm':
#         from .timm.model_timm import ModelTimm as ModelInterface
#     elif args.model_source == 'ultralytics':
#         from .ultralytics.model_ultralytics import ModelUltralytics as ModelInterface
#     else:
#         raise Exception(f'Unsupported model source: {args.model_source}')
#
#     return ModelInterface(args)


def get_library_name(lib):
    return 'benchmark.pytorch_code.' + lib + '.model_lib_' + lib, 'benchmark.pytorch_code.' + lib, 'benchmark.pytorch_code'


def find_modules(lib):
    # pytorch
    if importlib.util.find_spec('benchmark.pytorch_code.' + lib) is not None:
        model_lib_module = 'benchmark.pytorch_code.' + lib + '.model_lib_' + lib
        quant_module = 'benchmark.pytorch_code.quant'
    elif importlib.util.find_spec('benchmark.keras_code.' + lib) is not None:
        model_lib_module = 'benchmark.keras_code.' + lib + '.model_lib_' + lib
        quant_module = 'benchmark.keras_code.quant'
    else:
        raise Exception(f'Error: model library {lib} was not found')
    return model_lib_module, quant_module

def supported_datasets(model_interface):
    if model_interface == 'torchvision':
        dataset_list = ["IMAGENET"]
    elif model_interface == 'timm':
        dataset_list = ["IMAGENET"]
    elif model_interface == 'ultralytics':
        dataset_list = ["COCO"]
    else:
        raise Exception(f'Unsupported model source: {model_interface}')

    return dataset_list

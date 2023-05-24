
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
    return 'benchmark.libraries.' + lib + '.model_lib_' + lib

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

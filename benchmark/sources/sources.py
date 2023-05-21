
def get_model_source(args):
    # todo: add torchvision object detection notion (for ssd)
    if args.model_source == 'torchvision':
        from .torchvision.model_torchvision import ModelTorchvision as ModelConfig
    elif args.model_source == 'timm':
        from .timm.model_timm import ModelTimm as ModelConfig
    elif args.model_source == 'ultralytics':
        from .ultralytics.model_ultralytics import ModelUltralytics as ModelConfig
    else:
        raise Exception(f'Unsupported model source: {args.model_source}')

    return ModelConfig(args)


def supported_datasets(model_source):
    if model_source == 'torchvision':
        dataset_list = ["IMAGENET"]
    elif model_source == 'timm':
        dataset_list = ["IMAGENET"]
    elif model_source == 'ultralytics':
        dataset_list = ["COCO"]
    else:
        raise Exception(f'Unsupported model source: {model_source}')

    return dataset_list

def get_model_source(args):
    # todo: add torchvision object detection notion (for ssd)
    if args.model_source == 'torchvision':
        from .torchvision.model_torchvision import ModelTorchvision as ModelConfig
    elif args.model_source == 'timm':
        from .timm.model_timm import ModelTimm as ModelConfig
    elif args.model_source == 'ultralytics':
        from .ultralytics.model_yolo import ModelYolo as ModelConfig

    return ModelConfig(args)
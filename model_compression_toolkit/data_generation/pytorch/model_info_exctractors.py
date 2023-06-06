import torch
from torch import nn
from torchvision.ops import FrozenBatchNorm2d
from model_compression_toolkit.data_generation.pytorch.constants import DEVICE


class OrigBNStatsHolder(object):
    def __init__(self, model, eps=1e-6):
        self.bn_params = self.get_bn_params(model)
        self.eps = eps

    def get_bn_layer_names(self):
        return list(self.bn_params.keys())

    def get_mean(self, bn_layer_name):
        return self.bn_params[bn_layer_name][0]

    def get_var(self, bn_layer_name):
        return self.bn_params[bn_layer_name][1]

    def get_std(self, bn_layer_name):
        var = self.bn_params[bn_layer_name][1]
        std = torch.sqrt(var + self.eps)
        return std

    @staticmethod
    def get_bn_params(model):
        bn_params = {}
        for name, module in model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, FrozenBatchNorm2d)):
                mean = module.running_mean.detach().clone().flatten().to(DEVICE)
                var = module.running_var.detach().clone().flatten().to(DEVICE)
                bn_params.update({name: (mean, var)})
        return bn_params


class BatchNormInputHook(object):
    """
    Forward_hook used to extract the input of an intermediate batch norm layer.
    """
    def __init__(self, bn_layer_name):
        super(BatchNormInputHook, self).__init__()
        self.bn_layer_name = bn_layer_name
        self.input = None

    def hook(self, module, input, output):
        self.input = input[0]

    def clear(self):
        self.input = None


class ActivationExtractor(object):
    def __init__(self, model, bn_layer_types):
        print("Num. of BatchNorm layers = " + str(sum([1 if isinstance(layer, tuple(bn_layer_types)) else 0 for layer in model.modules()])))
        self.hooks = {}
        self.hook_handles = []
        for name, module in model.named_modules():
            if isinstance(module, tuple(bn_layer_types)):
                bn_hook = BatchNormInputHook(bn_layer_name=name)
                self.hooks.update({name: bn_hook})
                hook_handle = module.register_forward_hook(bn_hook.hook)
                self.hook_handles.append(hook_handle)

    def get_activation(self, layer_name):
        return self.hooks.get(layer_name)

    def clear(self):
        for hook in self.hooks:
            hook.clear()

    def remove(self):
        self.clear()
        for handle in self.hook_handles:
            handle.remove()



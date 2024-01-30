import torch.nn


def replace_2d_deg_module(model: torch.nn.Module, old_module: type, new_module: callable, get_config: callable):
    """
    Replaces a 2nd degree submodule in the given ultralytics model with a new module.

    Args:
        model (torch.nn.Module): The model to modify.
        old_module (type): The old module type to replace.
        new_module (callable): A function or callable that creates the new module.
        get_config (callable): A function or callable that retrieves the configuration for creating the new module.

    Returns:
        torch.nn.Module: The modified model.
    """
    for n, m in model.named_children():
        for name, c in m.named_children():
            if isinstance(c, old_module):
                l = new_module(get_config(c))
                setattr(l, 'f', c.f)
                setattr(l, 'i', c.i)
                setattr(l, 'type', c.type)
                setattr(m, name, l)
    return model
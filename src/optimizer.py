from copy import deepcopy
from torch.optim import SGD, Adam, AdamW, ASGD, Adamax, Adadelta, Adagrad, RMSprop
from utils.logger import print_log


def create_optimizer(cfg, model):
    kwargs = deepcopy(cfg['training']['optimizer'] or {})
    name = kwargs.pop('name')
    if 'texture' in kwargs:
        txt_kwargs = kwargs.pop('texture')
        named_parameters = list(model.named_parameters())
        params = [p for n, p in named_parameters if not n.startswith('texture')]
        txt_params = [p for n, p in named_parameters if n.startswith('texture')]
        optimizer = get_optimizer(name)([dict(params=params), dict(params=txt_params, **txt_kwargs)], **kwargs)
    else:
        optimizer = get_optimizer(name)(model.parameters(), **kwargs)
    print_log(f'Optimizer "{name}" init: kwargs={cfg["training"]["optimizer"]}')
    return optimizer


def get_optimizer(name):
    if name is None:
        name = 'sgd'
    return {
        'sgd': SGD,
        'adam': Adam,
        'adamw': AdamW,
        'asgd': ASGD,
        'adamax': Adamax,
        'adadelta': Adadelta,
        'adagrad': Adagrad,
        'rmsprop': RMSprop,
    }[name]

from toolz import valmap
import torch
from .logger import print_log


SQRT_EPS = 1e-6


def get_torch_device(gpu=None, verbose=False):
    if torch.cuda.is_available():
        device, nb_dev = torch.device(gpu) if gpu is not None else torch.device('cuda:0'), torch.cuda.device_count()
    else:
        device, nb_dev = torch.device("cpu"), None
    if verbose:
        print_log(f"Torch device state: device={device}, nb_dev={nb_dev}")
    return device


def torch_to(inp, device, non_blocking=False):
    nb = non_blocking  # set to True when doing distributed jobs
    if isinstance(inp, torch.Tensor):
        return inp.to(device, non_blocking=nb)
    elif isinstance(inp, (list, tuple)):
        return type(inp)(map(lambda t: t.to(device, non_blocking=nb) if isinstance(t, torch.Tensor) else t, inp))
    elif isinstance(inp, dict):
        return valmap(lambda t: t.to(device, non_blocking=nb) if isinstance(t, torch.Tensor) else t, inp)
    else:
        raise NotImplementedError


def signed_pow(t, exponent):
    return torch.sign(t) * (torch.abs(t).pow(exponent))


def safe_pow(t, exponent, eps=SQRT_EPS):
    return t.clamp(eps).pow(exponent)

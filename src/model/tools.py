from collections import OrderedDict
from toolz import keymap

import numpy as np
from pytorch3d.transforms import random_rotations
import torch
from torch import nn
from torch.nn import functional as F

from utils.logger import print_log


N_UNITS = 128
N_LAYERS = 3


def safe_model_state_dict(state_dict):
    """Convert a state dict saved from a DataParallel module to normal module state_dict."""
    if not next(iter(state_dict)).startswith("module."):
        return state_dict  # abort if dict is not a DataParallel model_state
    return keymap(lambda s: s[7:], state_dict, factory=OrderedDict)  # remove 'module.'


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def conv3x3(in_planes, out_planes, stride=1, padding=1, groups=1, dilation=1, zero_init=False):
    """3x3 convolution with padding"""
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, groups=groups, bias=False, dilation=dilation)
    if zero_init:
        conv.weight.data.zero_()
    return conv


def linear_normalize(tensor):
    M, m = tensor.max(), tensor.min()
    return (tensor - m) / (M - m)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def deconv4x4(in_planes, out_planes, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=stride, padding=padding)


def create_mlp(in_ch, out_ch, n_units=N_UNITS, n_layers=N_LAYERS, kaiming_init=True, zero_last_init=False,
               bias_last=True, with_norm=False, dropout=False):
    if n_layers > 0:
        seq = [nn.Linear(in_ch, n_units)]
        if with_norm:
            seq.append(nn.BatchNorm1d(n_units))
        seq.append(nn.ReLU(True))
        for _ in range(n_layers - 1):
            if dropout:
                seq.append(nn.Dropout(dropout))
            seq.append(nn.Linear(n_units, n_units))
            if with_norm:
                seq.append(nn.BatchNorm1d(n_units))
            seq.append(nn.ReLU(True))
        seq += [nn.Linear(n_units, out_ch, bias=bias_last)]
    else:
        seq = [nn.Linear(in_ch, out_ch, bias=bias_last)]
    mlp = nn.Sequential(*seq)

    if kaiming_init:
        mlp.apply(kaiming_weights_init)
    if zero_last_init:
        with torch.no_grad():
            if isinstance(zero_last_init, bool):
                mlp[-1].weight.zero_()
            else:  # We interpret as std
                mlp[-1].weight.normal_(mean=0, std=zero_last_init)
            if bias_last:
                mlp[-1].bias.zero_()
    return mlp


@torch.no_grad()
def kaiming_weights_init(m, nonlinearity='relu'):
    if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight, nonlinearity=nonlinearity)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def get_nb_out_channels(layer):
    last_module = list(filter(lambda e: isinstance(e, (nn.Conv2d, nn.Linear)), layer.modules()))[-1]
    if isinstance(last_module, nn.Conv2d):
        return last_module.out_channels
    else:
        return last_module.out_features


def get_output_size(in_channels, img_size, model):
    x = torch.zeros(1, in_channels, *img_size)
    return np.prod(model(x).shape)


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x


##########################################
# Generator utils
##########################################


class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        kernel = torch.Tensor([1, 2, 1])
        kernel = kernel[None, None, :] * kernel[None, :, None]
        kernel = kernel / kernel.norm(p=1)
        self.register_buffer('kernel', kernel)

    def forward(self, x):
        B, C, H, W = x.shape
        x = F.pad(x, (1, 1, 1, 1), mode='reflect')
        kernel = self.kernel.unsqueeze(1).expand(-1, C, -1, -1)
        kernel = kernel.reshape(-1, 1, *kernel.shape[2:])
        x = x.view(-1, kernel.size(0), x.size(-2), x.size(-1))
        return F.conv2d(x, kernel, groups=kernel.size(0), padding=0, stride=1).view(B, C, H, W)


def create_upsample_layer(name):
    if name == 'nn':
        return nn.Upsample(scale_factor=2)
    elif name == 'bilinear':
        return nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    elif name == 'bilinear_blur':
        return nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), Blur())
    else:
        raise NotImplementedError


##########################################
# Rendering / pose utils
##########################################


def init_rotations(init_type='uniform', N=None, n_elev=None, n_azim=None, elev_range=None, azim_range=None):
    if init_type == 'uniform':
        assert n_elev is not None and n_azim is not None
        assert N == n_elev * n_azim if N is not None else True
        eb, ee = elev_range if elev_range is not None else (-90, 90)
        ab, ae = azim_range if azim_range is not None else (-180, 180)
        er, ar = ee - eb, ae - ab
        elev = torch.Tensor([k*er/n_elev + eb - er/(2*n_elev) for k in range(1, n_elev + 1)])  # [-60, 0, 60]
        if ar == 360 and n_azim > 1:  # need a special case to avoid duplicated init at -180/180
            azim = torch.Tensor([k*ar/n_azim + ab for k in range(n_azim)])  # e.g. [-180, -90, 0, 90]
        else:
            azim = torch.Tensor([k*ar/n_azim + ab - ar/(2*n_azim) for k in range(1, n_azim + 1)])  # [-60, 0, 60]
        elev, azim = map(lambda t: t.flatten(), torch.meshgrid(elev, azim, indexing='ij'))
        roll = torch.zeros(elev.shape)
        print_log(f'init_rotations: azim={azim.tolist()}, elev={elev.tolist()}, roll={roll.tolist()}')
        R_init = torch.stack([azim, elev, roll], dim=1)
    elif init_type.startswith('random'):
        R_init = random_rotations(N)
    else:
        raise NotImplementedError
    return R_init


def azim_to_rotation_matrix(azim, as_degree=True):
    """Angle with +X in XZ plane"""
    if isinstance(azim, (int, float)):
        azim = torch.Tensor([azim])
    azim_rad = azim * np.pi / 180 if as_degree else azim
    R = torch.eye(3, device=azim.device)[None].repeat(len(azim), 1, 1)
    cos, sin = torch.cos(azim_rad), torch.sin(azim_rad)
    zeros = torch.zeros(len(azim), device=azim.device)
    R[:, 0, :] = torch.stack([cos, zeros, sin], dim=-1)
    R[:, 2, :] = torch.stack([-sin, zeros, cos], dim=-1)
    return R.squeeze()


def elev_to_rotation_matrix(elev, as_degree=True):
    """Angle with +Z in YZ plane"""
    if isinstance(elev, (int, float)):
        elev = torch.Tensor([elev])
    elev_rad = elev * np.pi / 180 if as_degree else elev
    R = torch.eye(3, device=elev.device)[None].repeat(len(elev), 1, 1)
    cos, sin = torch.cos(-elev_rad), torch.sin(-elev_rad)
    R[:, 1, 1:] = torch.stack([cos, sin], dim=-1)
    R[:, 2, 1:] = torch.stack([-sin, cos], dim=-1)
    return R.squeeze()


def roll_to_rotation_matrix(roll, as_degree=True):
    """Angle with +X in XY plane"""
    if isinstance(roll, (int, float)):
        roll = torch.Tensor([roll])
    roll_rad = roll * np.pi / 180 if as_degree else roll
    R = torch.eye(3, device=roll.device)[None].repeat(len(roll), 1, 1)
    cos, sin = torch.cos(roll_rad), torch.sin(roll_rad)
    R[:, 0, :2] = torch.stack([cos, sin], dim=-1)
    R[:, 1, :2] = torch.stack([-sin, cos], dim=-1)
    return R.squeeze()


def cpu_angle_between(R1, R2, as_degree=True):
    angle = ((torch.einsum('bii -> b', (R1.transpose(-2, -1) @ R2).view(-1, 3, 3)) - 1) / 2).acos()
    return 180 / np.pi * angle if as_degree else angle

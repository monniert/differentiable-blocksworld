from math import exp
import lpips
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models as tvmodels

from utils.chamfer import chamfer_distance
from utils.pytorch import safe_pow


def get_loss(name):
    return {
        'bce': nn.BCEWithLogitsLoss,
        'mse': nn.MSELoss,
        'l2': nn.MSELoss,
        'l1': nn.L1Loss,
        'huber': nn.SmoothL1Loss,
        'cosine': nn.CosineSimilarity,
        'perceptual': PerceptualLoss,
        'lpips': LPIPSLoss,
        'ssim': SSIMLoss,
        'chamfer': ChamferLoss,
        'tv': TotalVariationLoss,
    }[name]


def mse2psnr(x):
    return -10. * torch.log(x) / torch.log(torch.Tensor([10.]).to(x.device))


class LPIPSLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = lpips.LPIPS(net='vgg', verbose=False)
        for param in self.loss.parameters():
            param.requires_grad = False

    def forward(self, imgs, rec):
        return self.loss(imgs, rec, normalize=True).mean()


tv_norm_funcs = {
    'l1': lambda t: t.abs().sum(-1),
    'l2': lambda t: safe_pow(t.pow(2).sum(-1), 0.5),  # we use safe_pow to avoid gradient overflow at 0
    'l2sq': lambda t: t.pow(2).sum(-1),
}


class TotalVariationLoss(nn.Module):
    def forward(self, imgs):
        # imgs  of size BCHW
        dx, dy = torch.diff(imgs, dim=3).pow(2), torch.diff(imgs, dim=2).pow(2)
        return (dx + dy).mean()


class ChamferLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, x, y):
        return chamfer_distance(x, y, batch_reduction=self.reduction)


class PerceptualLoss(nn.Module):
    def __init__(self, normalize_input=True, normalize_features=True, feature_levels=None, sum_channels=False,
                 requires_grad=False):
        super().__init__()
        self.normalize_input = normalize_input
        self.normalize_features = normalize_features
        self.sum_channels = sum_channels
        self.feature_levels = feature_levels if feature_levels is not None else [3]
        assert isinstance(self.feature_levels, (list, tuple))
        self.max_level = max(self.feature_levels)
        self.register_buffer('mean_rgb', torch.Tensor([0.485, 0.456, 0.406]))
        self.register_buffer('std_rgb', torch.Tensor([0.229, 0.224, 0.225]))

        layers = tvmodels.vgg16(weights=tvmodels.VGG16_Weights.IMAGENET1K_V1).features
        self.slice1 = layers[:4]     # relu1_2
        self.slice2 = layers[4:9]    # relu2_2
        self.slice3 = layers[9:16]   # relu3_3
        self.slice4 = layers[16:23]  # relu4_3
        self.slice5 = layers[23:30]  # relu5_3
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, imgs, rec):
        inp = torch.cat([imgs, rec], 0)
        if self.normalize_input:
            inp = (inp - self.mean_rgb.view(1, 3, 1, 1)) / self.std_rgb.view(1, 3, 1, 1)

        feats = []
        for k in range(1, 6):
            if k > self.max_level:
                break
            inp = getattr(self, f'slice{k}')(inp)
            feats.append(torch.chunk(inp, 2, dim=0))

        losses = []
        for k, (f1, f2) in enumerate(feats, start=1):
            if k in self.feature_levels:
                if self.normalize_features:
                    f1, f2 = map(lambda t: t / (t.norm(dim=1, keepdim=True) + 1e-10), [f1, f2])
                loss = (f1 - f2) ** 2
                if self.sum_channels:
                    losses.append(loss.sum(1).flatten(2).mean(2))
                else:
                    losses.append(loss.flatten(1).mean(1))
        return sum(losses).mean()


######################################################################
# SSIM original repo implem: https://github.com/Po-Hsun-Su/pytorch-ssim
######################################################################


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, channel=3, padding=True):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.window = self.create_window(window_size, channel)
        self.padding = window_size // 2 if padding else 0

    def ssim(self, img1, img2):
        channel = self.channel
        window = self.window.to(img1.device)
        mu1 = F.conv2d(img1, window, padding=self.padding, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.padding, groups=channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.padding, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.padding, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.padding, groups=channel) - mu1_mu2
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1 - ssim_map

    @staticmethod
    def create_window(window_size, channel):
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, inp, target):
        return self.ssim(inp, target).flatten(1).mean(1)

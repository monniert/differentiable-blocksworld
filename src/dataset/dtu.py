from copy import deepcopy
from functools import cached_property
from PIL import Image

import cv2
import numpy as np
from pytorch3d.io import load_ply
import torch
from torch.utils.data.dataset import Dataset as TorchDataset
from torchvision.transforms import ToTensor, Compose, Resize

from utils import path_exists, get_files_from, use_seed
from utils.image import IMG_EXTENSIONS
from utils.path import DATASETS_PATH


EVAL_SCAN_IDS = [f'scan{i}' for i in [24, 31, 40, 45, 55, 59, 63, 75, 83, 105]]


class DTUDataset(TorchDataset):
    name = 'dtu'
    raw_img_size = (1200, 1600)
    n_channels = 3

    def __init__(self, split, img_size, tag, **kwargs):
        kwargs = deepcopy(kwargs)
        self.split = split
        self.tag = tag
        self.data_path = path_exists(DATASETS_PATH / 'DTU' / tag / 'image')
        self.input_files = get_files_from(self.data_path, IMG_EXTENSIONS, recursive=True, sort=True)
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        N = len(self.input_files)
        view_ids = kwargs.pop('view_ids', list(range(N)))
        self.view_ids = list(range(N)) if split == 'test' else view_ids
        self.on_disk = kwargs.pop('on_disk', False)
        assert len(kwargs) == 0, kwargs

        if split != 'train':
            with use_seed(len(split + tag)):
                np.random.shuffle(self.view_ids)

        cam = np.load(self.data_path.parent / 'cameras.npz')
        proj_mat = [(cam[f'world_mat_{i}'] @ cam[f'scale_mat_{i}'])[:3, :4] for i in range(N)]
        self.KRT = [pytorch3d_KRT_from_proj(p, image_size=self.raw_img_size) for p in proj_mat]

        filename = 'stl{}_total.ply'.format(tag.replace('scan', '').zfill(3))
        points = load_ply(self.data_path.parent.parent / 'Points' / 'stl' / filename)[0]
        self.scale_mat = torch.from_numpy(cam['scale_mat_0'])
        scale_inv = self.scale_mat.inverse()
        self.pc_gt = points @ scale_inv[:3, :3] + scale_inv[:3, 3]

        if self.on_disk:
            self.imgs = [self.transform(Image.open(f).convert('RGB')) for f in self.input_files]

    def __len__(self):
        return len(self.view_ids)

    def __getitem__(self, i):
        idx = self.view_ids[i]
        if self.on_disk:
            imgs = self.imgs[idx]
        else:
            imgs = self.transform(Image.open(self.input_files[idx]).convert('RGB'))
        K, R, T = self.KRT[idx]
        out = {'imgs': imgs, 'K': K, 'R': R, 'T': T}
        indices = torch.randperm(len(self.pc_gt))[:int(1e5)]
        pc = self.pc_gt[indices]
        return out, {'points': pc}

    @cached_property
    def transform(self):
        return Compose([Resize(self.img_size), ToTensor()])


def pytorch3d_KRT_from_proj(P, image_size):
    K, R, T = map(torch.from_numpy, opencv_KRT_from_proj(P))
    # DTU convention is x_world = R @ x_cam + T, PyTorch3D convention is x_cam = R_p @ x_world + T_p
    # we have x_cam = R.T @ x_world - R.T @ T, thus R_p = R.T and T_p = - R.T @ T
    R = R.T
    T = - R @ T

    # Pytorch3d from opencv, adapted from _cameras_from_opencv_projection in pytorch3d/renderer/camera_conversions.py
    #####################################

    # Retype the image_size correctly and flip to width, height.
    if isinstance(image_size, (tuple, list)):
        image_size = torch.Tensor(image_size)[None]
    image_size_wh = image_size.to(R).flip(dims=(1,))

    # Screen to NDC conversion:
    # For non square images, we scale the points such that smallest side
    # has range [-1, 1] and the largest side has range [-u, u], with u > 1.
    # This convention is consistent with the PyTorch3D renderer, as well as
    # the transformation function `get_ndc_to_screen_transform`.
    scale = image_size_wh.to(R).min(dim=1, keepdim=True)[0] / 2.0
    scale = scale.expand(-1, 2)
    c0 = image_size_wh / 2.0

    # Get the PyTorch3D focal length and principal point.
    focal_pytorch3d = torch.stack([K[0, 0], K[1, 1]], dim=-1) / scale
    p0_pytorch3d = -(K[:2, 2] - c0) / scale
    K_pytorch3d = torch.zeros(K.shape)
    K_pytorch3d[0, 0] = focal_pytorch3d[..., 0]
    K_pytorch3d[1, 1] = focal_pytorch3d[..., 1]
    K_pytorch3d[:2, 2] = p0_pytorch3d
    K_pytorch3d[2:, 2:] = 1 - torch.eye(2)

    # For R, T we flip x, y axes (opencv screen space has an opposite
    # orientation of screen axes).
    # We also transpose R (opencv multiplies points from the opposite=left side).
    R_pytorch3d = R.clone().T
    T_pytorch3d = T.clone()
    R_pytorch3d[:, :2] *= -1
    T_pytorch3d[:2] *= -1
    return K_pytorch3d, R_pytorch3d, T_pytorch3d


def opencv_KRT_from_proj(P):
    K_raw, R, T = cv2.decomposeProjectionMatrix(P)[:3]
    K = np.eye(4, dtype=np.float32)
    K[:3, :3] = K_raw / K_raw[2, 2]
    R = R.T
    T = (T[:3] / T[3])[:, 0]  # see https://stackoverflow.com/questions/62686618/opencv-decompose-projection-matrix
    return K, R, T

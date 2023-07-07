from copy import deepcopy
from functools import cached_property
from PIL import Image

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.exporter.exporter_utils import generate_point_cloud
import numpy as np
from pytorch3d.renderer.cameras import _get_sfm_calibration_matrix
import torch
from torch.utils.data.dataset import Dataset as TorchDataset
from torchvision.transforms import ToTensor, Compose, Resize

from utils import path_exists, use_seed
from utils.path import DATASETS_PATH


MODELS = {
    'campanile': 'data-nerfstudio-campanile/nerfacto/2023-01-05_153950',
    'one_side_cylinders': 'data-nerfstudio-one_side_cylinders/nerfacto/2023-05-15_143532',
}


class NerfstudioDataset(TorchDataset):
    name = 'nerfstudio'
    raw_img_size = (540, 960)
    bounding_box_min = (-4, -4, -4)
    bounding_box_max = (4, 4, 4)

    def __init__(self, split, tag, **kwargs):
        kwargs = deepcopy(kwargs)
        self.split = split
        self.tag = tag
        self.data_path = path_exists(DATASETS_PATH / 'nerfstudio' / 'data' / 'nerfstudio' / tag)
        self.model_path = path_exists(DATASETS_PATH / 'nerfstudio' / 'outputs' / MODELS[tag])
        if split != 'val':
            _, pipeline, _ = eval_setup(self.model_path / 'config.yml')
            if split == 'train':
                dataset = pipeline.datamanager.train_dataset
            else:
                dataset = pipeline.datamanager.eval_dataset

            self.input_files = dataset._dataparser_outputs.image_filenames
            print(self.input_files[0])
            self.N = len(self.input_files)
            self.view_ids = list(range(self.N))
            if split == 'test':
                with use_seed(len(tag)):
                    np.random.shuffle(self.view_ids)

            self.cameras = dataset._dataparser_outputs.cameras
            self.downscale_factor = kwargs.pop('downscale_factor', 1)
            if self.downscale_factor == 1:
                self.img_size = self.raw_img_size
            else:
                self.img_size = tuple(map(lambda x: round(x / self.downscale_factor), self.raw_img_size))
                self.cameras.rescale_output_resolution(1 / self.downscale_factor)
            assert len(kwargs) == 0, kwargs

            # Pytorch3d-compatible camera matrices
            # Intrinsics
            cx, cy, fx, fy = map(lambda x: getattr(self.cameras, x)[0].item(), ['cx', 'cy', 'fx', 'fy'])
            image_size = torch.Tensor(self.img_size[::-1],)[None]
            scale = image_size.min(dim=1, keepdim=True)[0] / 2.0
            c0 = image_size / 2.0
            p0_pytorch3d = -(torch.Tensor((cx, cy),)[None] - c0) / scale
            focal_pytorch3d = torch.Tensor([fx, fy])[None] / scale
            K = _get_sfm_calibration_matrix(1, 'cpu', focal_pytorch3d, p0_pytorch3d, orthographic=False)
            self.K = K.expand(self.N, -1, -1)

            # Extrinsics
            line = torch.Tensor([[0., 0., 0., 1.]]).expand(self.N, -1, -1)
            cam2world = torch.cat([self.cameras.camera_to_worlds, line], dim=1)
            self.cam2world = cam2world
            self.world2cam = cam2world.inverse()
            R, T = self.world2cam.split([3, 1], dim=-1)
            self.R = R[:, :3].transpose(1, 2) * torch.Tensor([-1., 1., -1])
            self.T = T.squeeze(2)[:, :3] * torch.Tensor([-1., 1., -1])

            self.pc_gt = self.generate_gt(pipeline)[:, :3]

    def generate_gt(self, pipeline):
        pcd = generate_point_cloud(pipeline=pipeline, bounding_box_min=self.bounding_box_min,
                                   bounding_box_max=self.bounding_box_max)
        points = torch.from_numpy(np.asarray(pcd.points, dtype=np.float32))
        points = torch.cat([points, torch.ones(points.shape[0], 1)], dim=-1)
        return points

    def __len__(self):
        return self.N if self.split != 'val' else 0

    def __getitem__(self, j):
        i = self.view_ids[j]
        imgs = self.transform(Image.open(self.input_files[i]).convert('RGB'))
        indices = torch.randperm(len(self.pc_gt))[:int(1e5)]
        pc = self.pc_gt[indices]
        out = {'imgs': imgs, 'K': self.K[i], 'R': self.R[i], 'T': self.T[i]}
        return out, {'points': pc}

    @cached_property
    def transform(self):
        return Compose([Resize(self.img_size), ToTensor()])

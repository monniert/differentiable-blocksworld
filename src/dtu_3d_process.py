import argparse

import numpy as np
from pytorch3d.io import save_ply
import torch

from dataset import get_dataset
from dataset.dtu import EVAL_SCAN_IDS
from utils import use_seed, path_mkdir
from utils.logger import create_logger, print_log
from utils.path import EMS_PATH, MBF_PATH
from utils.pytorch import get_torch_device
from utils.ransac import Ransac


N_POINTS_EMS_FIT = 5000
N_POINTS_MBF_FIT = 200000


class DTU3DPreprocess:
    """Pipeline to preprocess DTU 3D GT for 3D baseline models"""
    def __init__(self, run_dir, model_name, filter_ground=False):
        self.run_dir = path_mkdir(run_dir)
        self.model_name = model_name
        self.filter_ground = filter_ground
        self.device = get_torch_device(verbose=True)
        print_log(f'{self.__class__.__name__} init: run_dir={run_dir}, filter_ground={filter_ground}')

    @use_seed()
    def run(self):
        for tag in EVAL_SCAN_IDS:
            print_log(f'Preprocessing and saving GT point cloud for {tag}...')
            dataset = get_dataset('dtu')(split='train', img_size=(300, 400), tag=tag)
            pc = dataset.pc_gt[torch.randperm(len(dataset.pc_gt))].to(self.device)
            scale_mat = dataset.scale_mat.to(self.device)
            if self.filter_ground:
                ransac = Ransac()
                ransac.fit(pc[:, :2], pc[:, 2:3])
                model = ransac.best_models[-1]
                is_inlier = (model.predict(pc[:, :2]) - pc[:, 2:]).pow(2) < ransac.thresh
                pc = pc[~is_inlier[:, 0]]

            # Back in original space
            pc = pc @ scale_mat[:3, :3] + scale_mat[:3, 3]
            if self.model_name == 'ems':
                n_points = N_POINTS_EMS_FIT
                # We standardize data following EMS practices
                mean = pc.mean(0)
                pc = pc - mean
                scale = 10 / pc.max()
                pc = pc * scale
            elif self.model_name == 'mbf':
                n_points = N_POINTS_MBF_FIT
                # We standardize data following MBF practices
                mean = pc.mean(0)
                pc = pc - mean
                scale = 5 / pc.max()
                pc = pc * scale
            else:
                raise NotImplementedError

            np.save(self.run_dir / f'{tag}_scale.npy', torch.cat([mean, scale.expand(1)]).cpu().numpy())
            save_ply(self.run_dir / f'{tag}.ply', pc[:n_points].cpu())

            print_log('Done')
        print_log(f'{self.__class__.__name__} over')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pipeline to preprocess DTU 3D GT for 3D baseline models')
    parser.add_argument('-n', '--name', nargs='?', type=str, required=True, help='Name of the baseline model')
    parser.add_argument('-t', '--tag', nargs='?', type=str, required=True, help='Run tag of the experiment')
    parser.add_argument('-f', '--filter_ground', action='store_true')
    parser.add_argument('-s', '--seed', nargs='?', type=int, default=1234, help='Seed')
    args = parser.parse_args()
    assert (args.name in ['ems', 'mbf']) & (args.tag != '')

    path = EMS_PATH if args.name == 'ems' else MBF_PATH
    run_dir = path_mkdir(path / 'dtu' / args.tag)
    create_logger(run_dir, name='3d_process')
    evaluator = DTU3DPreprocess(run_dir, model_name=args.name, filter_ground=args.filter_ground)
    evaluator.run(seed=args.seed)

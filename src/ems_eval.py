import argparse
from collections import OrderedDict
import shutil

import numpy as np
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures.meshes import join_meshes_as_scene, Meshes
from scipy.spatial.transform import Rotation
import torch

from dataset import get_dataset
from dataset.dtu import EVAL_SCAN_IDS
from utils import use_seed, path_mkdir, path_exists
from utils.chamfer import chamfer_distance
from utils.dtu_eval import evaluate_mesh
from utils.logger import create_logger, print_log
from utils.path import RUNS_PATH, EMS_PATH, DATASETS_PATH
from utils.pytorch import get_torch_device
from utils.superquadric import create_sq_meshes


N_POINTS_EVAL = int(5e5)
CHAMFER_FACTOR = 10


class EMSEvaluator:
    """Pipeline to evaluate EMS results on DTU"""
    def __init__(self, run_dir, ems_tag=None):
        self.run_dir = path_mkdir(run_dir)
        ems_dir = path_exists(EMS_PATH / 'dtu' / (ems_tag or self.run_dir.name))
        shutil.copytree(str(ems_dir), str(self.run_dir), dirs_exist_ok=True)
        self.device = get_torch_device(verbose=True)
        print_log(f'{self.__class__.__name__} init: run_dir={run_dir})')

    @use_seed()
    def run(self):
        device = self.device
        for tag in EVAL_SCAN_IDS:
            print_log(f'Evaluate EMS algorithm for {tag}...')
            dataset = get_dataset('dtu')(split='train', img_size=(300, 400), tag=tag)
            gt = dataset.pc_gt[torch.randperm(len(dataset.pc_gt))][:N_POINTS_EVAL]

            # Create mesh
            mean, scale_ems = torch.from_numpy(np.load(self.run_dir / f'{tag}_scale.npy')).split([3, 1])
            params = torch.from_numpy(np.load(self.run_dir / f'{tag}_spq.npy')).float()
            N = len(params)
            eps1, eps2, S, T = params[:, 0:1], params[:, 1:2], params[:, 2:5], params[:, 8:11]
            R = torch.from_numpy(Rotation.from_euler('ZYX', params[:, 5:8]).as_matrix()).float()
            meshes = create_sq_meshes(eps1, eps2, S, level=1)
            verts = (R @ meshes.verts_padded().permute(0, 2, 1) + T[:, :, None]).permute(0, 2, 1)
            # we rescale verts to original space
            verts = verts / scale_ems + mean
            scene = join_meshes_as_scene(Meshes(verts, meshes.faces_padded()))

            # Custom chamfer-L1 evaluation using VolSDF scaling
            points = sample_points_from_meshes(scene, num_samples=N_POINTS_EVAL).to(device)
            scale_inv = dataset.scale_mat.inverse().to(device)
            points = points @ scale_inv[:3, :3] + scale_inv[:3, 3]  # do VolSDF standardisation
            gt = gt[None].to(device)
            acc, comp = chamfer_distance(points, gt, return_L1=True, direction_reduction='none')[0]
            acc, comp = CHAMFER_FACTOR * acc.item(), CHAMFER_FACTOR * comp.item()

            scores = OrderedDict([('n_blocks', N), ('chL1_acc', acc), ('chL1_comp', comp)])
            print_log(f'{tag}_scores: ' + ', '.join(["{}={:.5f}".format(k, v) for k, v in scores.items()]))
            with open(self.run_dir / f'{tag}_scores.tsv', mode='w') as f:
                f.write("\t".join(scores.keys()) + "\n")
                f.write("\t".join(map('{:.5f}'.format, scores.values())) + "\n")

            # Official DTU evaluation
            evaluate_mesh(scene, int(tag.replace('scan', '')), DATASETS_PATH / 'DTU', self.run_dir,
                          suffix=f'_{tag}', save_viz=False)

            print_log('Done')

        print_log(f'{self.__class__.__name__} over')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pipeline to evaluate EMS results on DTU')
    parser.add_argument('-t', '--tag', nargs='?', type=str, required=True, help='Run tag of the experiment')
    parser.add_argument('-e', '--ems_tag', nargs='?', type=str, help='EMS tag of the experiment')
    parser.add_argument('-s', '--seed', nargs='?', type=int, default=1234, help='Seed')
    args = parser.parse_args()
    assert args.tag != ''

    run_dir = path_mkdir(RUNS_PATH / 'ems' / args.tag)
    create_logger(run_dir, name='ems_eval')
    evaluator = EMSEvaluator(run_dir, args.ems_tag or args.tag)
    evaluator.run(seed=args.seed)

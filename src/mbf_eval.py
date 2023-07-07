import argparse
from collections import OrderedDict
import json
import shutil

import numpy as np
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures.meshes import join_meshes_as_scene, Meshes
import torch
import trimesh

from dataset import get_dataset
from dataset.dtu import EVAL_SCAN_IDS
from utils import use_seed, path_mkdir, path_exists
from utils.chamfer import chamfer_distance
from utils.dtu_eval import evaluate_mesh
from utils.logger import create_logger, print_log
from utils.path import RUNS_PATH, MBF_PATH, DATASETS_PATH
from utils.pytorch import get_torch_device


N_POINTS_EVAL = int(5e5)
CHAMFER_FACTOR = 10


class MBFEvaluator:
    """Pipeline to evaluate MBF results on DTU"""
    def __init__(self, run_dir, mbf_tag=None):
        self.run_dir = path_mkdir(run_dir)
        mbf_dir = path_exists(MBF_PATH / 'dtu' / (mbf_tag or self.run_dir.name))
        shutil.copytree(str(mbf_dir), str(self.run_dir), dirs_exist_ok=True)
        self.device = get_torch_device(verbose=True)
        print_log(f'{self.__class__.__name__} init: run_dir={run_dir}')

    @use_seed()
    def run(self):
        device = self.device
        for tag in EVAL_SCAN_IDS:
            print_log(f'Evaluate MBF algorithm for {tag}...')
            dataset = get_dataset('dtu')(split='train', img_size=(300, 400), tag=tag)
            gt = dataset.pc_gt[torch.randperm(len(dataset.pc_gt))][:N_POINTS_EVAL]

            # Creating mesh
            mean, scale_mbf = torch.from_numpy(np.load(self.run_dir / f'{tag}_scale.npy')).split([3, 1])
            with open(self.run_dir / tag / 'UH.json', mode='r') as f:
                metrics = json.load(f)
            verts = torch.Tensor(metrics['bbox']).float()
            # we rescale verts to original space
            verts = verts / scale_mbf + mean
            N = len(verts)
            faces = torch.stack([torch.from_numpy(trimesh.convex.convex_hull(verts[k]).faces) for k in range(N)])
            scene = join_meshes_as_scene(Meshes(verts, faces))

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
    parser = argparse.ArgumentParser(description='Pipeline to evaluate MBF results on DTU')
    parser.add_argument('-t', '--tag', nargs='?', type=str, required=True, help='Run tag of the experiment')
    parser.add_argument('-e', '--mbf_tag', nargs='?', type=str, help='MBF tag of the experiment')
    parser.add_argument('-s', '--seed', nargs='?', type=int, default=1234, help='Seed')
    args = parser.parse_args()
    assert args.tag != ''

    run_dir = path_mkdir(RUNS_PATH / 'mbf' / args.tag)
    create_logger(run_dir, name='mbf_eval')
    evaluator = MBFEvaluator(run_dir, args.mbf_tag or args.tag)
    evaluator.run(seed=args.seed)

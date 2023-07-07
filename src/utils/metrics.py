from collections import defaultdict, OrderedDict
import pandas as pd
from pathlib import Path
from pytorch3d.ops import sample_points_from_meshes as sample_points, iterative_closest_point as torch_icp
from pytorch3d.structures import Meshes
import torch

from .chamfer import chamfer_distance
from .logger import print_log
from .mesh import voxelize, normalize


EPS = 1e-7
CHAMFER_FACTOR = 10  # standard multiplicative factor to report Chamfer, see OccNet or DVR


class AverageMeter:
    """Compute and store the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.avg = 0
        self.count = 0

    def update(self, val, N=1):
        if isinstance(val, torch.Tensor):
            assert val.numel() == 1
            val = val.item()
        self.val = val
        self.sum += val * N
        self.count += N
        self.avg = self.sum / self.count if self.count != 0 else 0


class Metrics:
    log_data = True

    def __init__(self, *names, log_file=None, append=False):
        self.names = list(names)
        self.meters = defaultdict(AverageMeter)
        if log_file is not None and self.log_data:
            self.log_file = Path(log_file)
            if not self.log_file.exists() or not append:
                with open(self.log_file, mode='w') as f:
                    f.write("iteration\tepoch\tbatch\t" + "\t".join(self.names) + "\n")
        else:
            self.log_file = None

    def log_and_reset(self, *names, it=None, epoch=None, batch=None):
        self.log(it, epoch, batch)
        self.reset(*names)

    def log(self, it, epoch, batch):
        if self.log_file is not None:
            with open(self.log_file, mode="a") as file:
                file.write(f"{it}\t{epoch}\t{batch}\t" + "\t".join(map("{:.6f}".format, self.values)) + "\n")

    def reset(self, *names):
        if len(names) == 0:
            names = self.names
        for name in names:
            self[name].reset()

    def read_log(self):
        if self.log_file is not None:
            return pd.read_csv(self.log_file, sep='\t', index_col=0)
        else:
            return pd.DataFrame()

    def __getitem__(self, name):
        return self.meters[name]

    def __repr__(self):
        return ', '.join(['{}={:.4f}'.format(name, self[name].avg) for name in self.names])

    def __len__(self):
        return len(self.names)

    @property
    def values(self):
        return [self[name].avg for name in self.names]

    def update(self, *name_val, N=1):
        if len(name_val) == 1:
            d = name_val[0]
            assert isinstance(d, dict)
            for k, v in d.items():
                self.update(k, v, N=N)
        else:
            assert len(name_val) == 2
            name, val = name_val
            if name not in self.names:
                raise KeyError(f'{name} not in current metrics')
            if isinstance(val, (tuple, list)):
                self[name].update(val[0], N=val[1])
            else:
                self[name].update(val, N=N)

    def get_named_values(self, filter_fn=None):
        names, values = self.names, self.values
        if filter_fn is not None:
            zip_fn = lambda k_v: filter_fn(k_v[0])
            names, values = map(list, zip(*filter(zip_fn, zip(names, values))))
        return list(zip(names, values))


class MeshEvaluator:
    """
    Mesh evaluation class by computing similarity metrics between predicted mesh and GT.
    Code inspired from https://github.com/autonomousvision/differentiable_volumetric_rendering (see im2mesh/eval.py)
    """
    default_names = ['chamfer-L1', 'chamfer-L1-ICP', 'normal-cos', 'normal-cos-ICP', '3D-IoU', '3D-IoU-ICP']

    def __init__(self, names=None, log_file=None, run_icp=True, estimate_scale=True, anisotropic_scale=True,
                 icp_type='gradient', fast_cpu=False, append=False):
        self.names = names if names is not None else self.default_names
        self.metrics = Metrics(*self.names, log_file=log_file, append=append)
        self.run_icp = run_icp
        self.estimate_scale = estimate_scale
        self.ani_scale = anisotropic_scale
        self.icp_type = icp_type
        assert icp_type in ['normal', 'gradient']
        self.fast_cpu = fast_cpu
        self.N = 50000 if fast_cpu else 100000
        print_log('MeshEvaluator init: run_icp={}, estimate_scale={}, anisotropic_scale={}, icp_type={}, n_iter={}'
                  .format(run_icp, estimate_scale, anisotropic_scale, icp_type, self.n_iter))

    @property
    def n_iter(self):
        if self.icp_type == 'normal':
            return 10 if self.fast_cpu else 30
        else:
            return 30 if self.fast_cpu else 100

    def update(self, mesh_pred, labels):
        pc_gt, norm_gt = labels['points'], labels['normals']
        vox_gt = labels.get('voxels')
        res = self.evaluate(mesh_pred, pc_gt=pc_gt, norm_gt=norm_gt, vox_gt=vox_gt)
        self.metrics.update(res, N=len(mesh_pred))

    def evaluate(self, mesh_pred, pc_gt, norm_gt, vox_gt=None):
        assert abs(pc_gt.abs().max() - 0.5) < 0.01  # XXX GT should fit in the unit cube [-0.5, 0.5]^3
        pc_pred, norm_pred = sample_points(mesh_pred, self.N, return_normals=True)
        if self.N < len(pc_gt):
            idxs = torch.randperm(len(pc_gt))[:self.N]
            pc_gt, norm_gt = pc_gt[:, idxs], norm_gt[:, idxs]

        use_scale, ani_scale, n_iter = self.estimate_scale, self.ani_scale, self.n_iter
        results = []
        if self.run_icp:
            # Normalize mesh to be centered around 0 and fit inside the unit cube for better ICP
            mesh_pred = normalize(mesh_pred)
            pc_pred2, norm_pred2 = sample_points(mesh_pred, self.N, return_normals=True)
            if self.icp_type == 'normal':
                pc_pred_icp, RTs = torch_icp(pc_pred2, pc_gt, estimate_scale=use_scale, max_iterations=n_iter)[2:4]
            else:
                from .icp import gradient_icp
                pc_pred_icp, RTs = gradient_icp(pc_pred2, pc_gt, use_scale, ani_scale, lr=0.01, n_iter=n_iter)
            pc_preds, norm_preds, tags = [pc_pred, pc_pred_icp], [norm_pred, norm_pred2], ['', '-ICP']
        else:
            pc_preds, norm_preds, tags = [pc_pred], [norm_pred], ['']

        for pc, norm, tag in zip(pc_preds, norm_preds, tags):
            chamfer_L1, normal = chamfer_distance(pc_gt, pc, x_normals=norm_gt, y_normals=norm,
                                                  return_L1=True, return_mean=True)
            chamfer_L1 = chamfer_L1 * CHAMFER_FACTOR
            results += [('chamfer-L1' + tag, chamfer_L1.item()), ('normal-cos' + tag, 1 - normal.item())]

        if vox_gt is not None and '3D-IoU' in self.names:
            raise NotImplementedError('not implemented for batch processing')
            vox_gt, vox_pred = vox_gt.float(), voxelize(mesh_pred).float()
            iou = (vox_gt * vox_pred).sum() / (vox_gt + vox_pred).clamp(0, 1).sum()
            results += [('3D-IoU', iou.item())]
            if self.run_icp:
                R, T, s = RTs
                mesh_pred = Meshes(s * mesh_pred.verts_padded() @ R + T, mesh_pred.faces_padded())
                vox_pred = voxelize(mesh_pred)
                iou = (vox_gt * vox_pred).sum() / (vox_gt + vox_pred).clamp(0, 1).sum()
                results += [('3D-IoU-ICP', iou.item())]

        results = list(filter(lambda x: x[0] in self.names, results))
        return OrderedDict(results)

    def compute(self):
        return self.metrics.values

    def __repr__(self):
        return self.metrics.__repr__()

    def log_and_reset(self, it, epoch, batch):
        self.metrics.log_and_reset(it=it, epoch=epoch, batch=batch)

    def read_log(self):
        return self.metrics.read_log()


class ProxyEvaluator:
    default_names = ['mask_iou']

    def __init__(self, names=None, log_file=None, append=False):
        self.names = names if names is not None else self.default_names
        self.metrics = Metrics(*self.names, log_file=log_file, append=append)

    def update(self, mask_pred, mask_gt):
        for k in range(len(mask_pred)):
            self.metrics.update(self.evaluate(mask_pred[k], mask_gt[k]))

    def evaluate(self, mask_pred, mask_gt):
        results = []
        miou = (mask_pred * mask_gt).sum() / (mask_pred + mask_gt).clamp(0, 1).sum()
        results += [('mask_iou', miou.item())]
        results = list(filter(lambda x: x[0] in self.names, results))
        return OrderedDict(results)

    def compute(self):
        return self.metrics.values

    def __repr__(self):
        return self.metrics.__repr__()

    def log_and_reset(self, it, epoch, batch):
        self.metrics.log_and_reset(it=it, epoch=epoch, batch=batch)

    def read_log(self):
        return self.metrics.read_log()

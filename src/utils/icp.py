from pytorch3d.transforms import rotation_6d_to_matrix
import torch
from torch import nn
from torch.optim import Adam

from .chamfer import chamfer_distance
from .metrics import AverageMeter


@torch.enable_grad()
def gradient_icp(pc_pred, pc_gt, estimate_scale=True, anisotropic_scale=False, lr=0.01, n_iter=300, batch_size=None,
                 shared_params=False, verbose=False):
    device = pc_pred.device
    assert len(pc_pred.shape) == 3 and len(pc_pred.shape) == 3, 'expected points to be of shape (N, P, D)'
    assert len(pc_pred) == len(pc_gt)

    N_params = 1 if shared_params else len(pc_pred)
    R_6d = nn.Parameter(torch.Tensor([[1., 0., 0., 0., 1., 0.]]).repeat(N_params, 1).to(device))
    T = nn.Parameter(torch.zeros(N_params, 3).to(device))
    if estimate_scale:
        s = nn.Parameter(torch.ones(N_params, 3 if anisotropic_scale else 1).to(device))
        params = [R_6d, T, s]
    else:
        s = torch.ones(N_params, 3).to(device)
        params = [R_6d, T]

    loss_meter = AverageMeter()
    loss_min, argmin = 1e6, [rotation_6d_to_matrix(R_6d).detach(), T.detach(), s.detach()]
    opt = Adam(params=params, lr=lr)
    if batch_size is not None:
        assert shared_params, 'per-instance optimization not implemented for batch optimization'
        N, B = len(pc_pred), batch_size
        n_batch = (N - 1) // B + 1
        cur_iter = 0
        while cur_iter < n_iter:
            idx = torch.randperm(N)
            pc_pred, pc_gt = pc_pred[idx], pc_gt[idx]
            for k in range(n_batch):
                opt.zero_grad()
                pcp, pcg = pc_pred[k*B:(k+1)*B], pc_gt[k*B:(k+1)*B]
                loss = chamfer_distance(s * pcp @ rotation_6d_to_matrix(R_6d) + T, pcg, return_mean=True)[0]
                loss.backward()
                opt.step()
                loss_meter.update(loss.item(), N=len(pcp))
                if cur_iter % 10 == 0:
                    loss_val = loss_meter.avg
                    if loss_val < loss_min:
                        loss_min, argmin = loss_val, [rotation_6d_to_matrix(R_6d).detach(), T.detach(), s.detach()]
                        if verbose:
                            print(loss_val, 'save checkpoint')
                    elif verbose:
                        print(loss_val)
                    loss_meter.reset()

                cur_iter += 1
                if cur_iter == n_iter:
                    break

    else:
        for cur_iter in range(n_iter):
            opt.zero_grad()
            loss = chamfer_distance(s[:, None] * pc_pred @ rotation_6d_to_matrix(R_6d) + T[:, None], pc_gt)[0]
            loss.backward()
            opt.step()
            loss_meter.update(loss.item(), N=len(pc_pred))
            if cur_iter % 10 == 0:
                loss_val = loss_meter.avg
                if loss_val < loss_min:
                    loss_min, argmin = loss_val, [rotation_6d_to_matrix(R_6d).detach(), T.detach(), s.detach()]
                    if verbose:
                        print(loss_val, 'save checkpoint')
                elif verbose:
                    print(loss_val)
                loss_meter.reset()

    Rf, Tf, sf = argmin
    upd_pc_pred = sf[:, None] * pc_pred @ Rf + Tf[:, None]
    return upd_pc_pred, [Rf, Tf, sf]

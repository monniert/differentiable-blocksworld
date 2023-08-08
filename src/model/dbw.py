from collections import OrderedDict
from copy import deepcopy
from toolz import valfilter
from pathlib import Path

import numpy as np
from pytorch3d.io import save_ply
from pytorch3d.ops.subdivide_meshes import SubdivideMeshes
from pytorch3d.renderer import TexturesUV
from pytorch3d.structures.meshes import join_meshes_as_scene, join_meshes_as_batch
from pytorch3d.structures import Meshes
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d, random_rotations
import torch
import torch.nn as nn
import torch.nn.functional as F


from .loss import get_loss, mse2psnr, tv_norm_funcs
from .renderer import Renderer, get_circle_traj, save_trajectory_as_video, save_mesh_as_video, DIRECTION_LIGHT
from .tools import safe_model_state_dict, elev_to_rotation_matrix, azim_to_rotation_matrix, roll_to_rotation_matrix
from utils import use_seed, path_mkdir
from utils.image import convert_to_img
from utils.logger import print_warning
from utils.metrics import AverageMeter
from utils.mesh import get_icosphere, get_icosphere_uvs, save_mesh_as_obj, get_plane, point_to_uv_sphericalmap
from utils.plot import get_fancy_cmap
from utils.pytorch import torch_to, safe_pow
from utils.superquadric import parametric_sq, implicit_sq, sample_sq


VIZ_SIZE = 256
DECIMATE_FACTOR = 8
OVERLAP_N_POINTS = 1000
OVERLAP_N_BLOCKS = 1.95
OVERLAP_TEMPERATURE = 0.005


class DifferentiableBlocksWorld(nn.Module):
    name = 'dbw'

    def __init__(self, img_size, **kwargs):
        super().__init__()
        self._init_kwargs = deepcopy(kwargs)
        self._init_kwargs['img_size'] = img_size
        self._init_blocks(**kwargs.get('mesh', {}))
        self._init_renderer(img_size, **kwargs.get('renderer', {}))
        self._init_rend_optim(**kwargs.get('rend_optim', {}))
        self._init_loss(**kwargs.get('loss', {}))
        self.cur_epoch = 0

    @property
    def init_kwargs(self):
        return deepcopy(self._init_kwargs)

    def _init_blocks(self, **kwargs):
        self.n_blocks = kwargs.pop('n_blocks', 1)
        self.S_world = kwargs.pop('S_world', 1)
        elev, azim, roll = kwargs.pop('R_world', [0, 0, 0])
        R_world = (elev_to_rotation_matrix(elev) @ azim_to_rotation_matrix(azim) @ roll_to_rotation_matrix(roll))[None]
        T_world = torch.Tensor(kwargs.pop('T_world', [0., 0., 0.]))[None]
        self.register_buffer('R_world', R_world)
        self.register_buffer('T_world', T_world)
        self.z_far = kwargs.pop('z_far', 10)
        self.ratio_block_scene = kwargs.pop('ratio_block_scene', 1 / 4)
        self.txt_size = kwargs.pop('txt_size', 256)
        self.txt_bkg_upscale = kwargs.pop('txt_bkg_upscale', 1)
        self.scale_min = kwargs.pop('scale_min', 0.2)
        opacity_init = kwargs.pop('opacity_init', 0.5)
        T_range = kwargs.pop('T_range', [1, 1, 1])
        T_init_mode = kwargs.pop('T_init_mode', 'gauss')
        assert len(kwargs) == 0, kwargs

        # Build spherical background and planar ground
        self.bkg = get_icosphere(level=2, flip_faces=True).scale_verts_(self.z_far)
        self.register_buffer('bkg_verts_uvs', point_to_uv_sphericalmap(self.bkg.verts_packed()))
        self.ground = get_plane().scale_verts_(torch.Tensor([self.z_far, 1, self.z_far])[None])
        for k in range(3):
            self.ground = SubdivideMeshes()(self.ground)
        self.register_buffer('ground_verts_uvs', (self.ground.verts_packed()[:, [0, 2]] / self.z_far + 1) / 2)

        # Build primitive blocks
        block = get_icosphere(level=1)
        self.blocks = join_meshes_as_batch([block.scale_verts(self.ratio_block_scene) for _ in range(self.n_blocks)])
        self.sq_eps = nn.Parameter(torch.zeros(self.n_blocks, 2))
        verts = self.blocks.verts_padded() / self.ratio_block_scene
        self.register_buffer('sq_eta', torch.asin((verts[..., 1])))
        self.register_buffer('sq_omega', torch.atan2(verts[..., 0], verts[..., 2]))
        faces_uvs, verts_uvs = get_icosphere_uvs(level=1, fix_continuity=True, fix_poles=True)
        p_left = abs(int(np.floor(verts_uvs.min(0)[0][0].item() * self.txt_size)))
        p_right = int(np.ceil((verts_uvs.max(0)[0][0].item() - 1) * self.txt_size))
        verts_u = (verts_uvs[..., 0] * self.txt_size + p_left) / (self.txt_size + p_left + p_right)
        verts_uvs = torch.stack([verts_u, verts_uvs[..., 1]], dim=-1)
        self.txt_padding = p_left, p_right
        self.BNF = len(faces_uvs)
        self.register_buffer('block_faces_uvs', faces_uvs)
        self.register_buffer('block_verts_uvs', verts_uvs)

        # Initialize learnable pose parameters
        self.R_6d_ground = nn.Parameter(torch.Tensor([[1., 0., 0., 0., 1., 0.]]))
        self.T_ground = nn.Parameter(torch.Tensor([[0., -0.9 * T_range[1], 0.]]))
        N = self.n_blocks
        S_init = (torch.rand(N, 3) + 0.5 - self.scale_min).log()
        R_6d_init = matrix_to_rotation_6d(random_rotations(N))
        if T_init_mode == 'gauss':
            T_init = torch.randn(N, 3) / 2 * torch.Tensor(T_range)
        elif T_init_mode == 'uni':
            T_init = (2 * torch.rand(N, 3) - 1) * torch.Tensor(T_range)
        else:
            raise NotImplementedError
        self.S = nn.Parameter(S_init.clone())
        self.R_6d = nn.Parameter(R_6d_init.clone())
        self.T = nn.Parameter(T_init.clone())

        # Initialize learnable opacity and texture parameters
        self.alpha_logit = nn.Parameter(torch.logit(torch.ones(N) * opacity_init) + 1e-3)
        TS, txt_scale = self.txt_size, self.txt_bkg_upscale
        self.texture_bkg = nn.Parameter(torch.randn(1, TS * txt_scale, TS * txt_scale, 3) / 10)
        self.texture_ground = nn.Parameter(torch.randn(1, TS * txt_scale, TS * txt_scale, 3) / 10)
        self.textures = nn.Parameter(torch.randn(N, TS, TS, 3) / 10)

    def _init_rend_optim(self, **kwargs):
        # Basic
        self.opacity_noise = kwargs.pop('opacity_noise', False)
        self.decouple_rendering = kwargs.pop('decouple_rendering', False)
        self.coarse_learning = kwargs.pop('coarse_learning', True)
        self.decimate_txt = kwargs.pop('decimate_txt', False)
        self.decim_factor = kwargs.pop('decimate_factor', DECIMATE_FACTOR)
        self.kill_blocks = kwargs.pop('kill_blocks', False)
        assert len(kwargs) == 0, kwargs

    def _init_renderer(self, img_size, **kwargs):
        self.renderer = Renderer(img_size, **kwargs)
        kwargs['sigma'] = 5e-6
        self.renderer_fine = Renderer(img_size, **kwargs)
        kwargs['faces_per_pixel'] = 1
        kwargs['sigma'] = 0
        kwargs['detach_bary'] = False
        self.renderer_env = Renderer(img_size, **kwargs)
        kwargs['lights'] = {'name': 'directional', 'direction': [DIRECTION_LIGHT], 'ambient_color': [[0.7, 0.7, 0.7]],
                            'diffuse_color': [[0.4, 0.4, 0.4]], 'specular_color': [[0., 0., 0.]]}
        kwargs['shading_type'] = 'flat'
        kwargs['background_color'] = (1, 1, 1)
        self.renderer_light = Renderer(img_size, **kwargs)

    def _init_loss(self, **kwargs):
        loss_weights = {
            'rgb': kwargs.pop('rgb_weight', 1.0),
            'perceptual': kwargs.pop('perceptual_weight', 0),
            'parsimony': kwargs.pop('parsimony_weight', 0),
            'scale': kwargs.pop('scale_weight', 0),
            'tv': kwargs.pop('tv_weight', 0),
            'overlap': kwargs.pop('overlap_weight', 0),
        }
        name = kwargs.pop('name', 'mse')
        perceptual_name = kwargs.pop('perceptual_name', 'lpips')
        self.tv_norm = tv_norm_funcs[kwargs.pop('tv_type', 'l2sq')]
        assert len(kwargs) == 0, kwargs

        self.loss_weights = valfilter(lambda v: v > 0, loss_weights)
        self.loss_names = [f'loss_{n}' for n in list(self.loss_weights.keys()) + ['total']]
        self.criterion = get_loss(name)()
        if 'perceptual' in self.loss_weights:
            self.perceptual_loss = get_loss(perceptual_name)()

    def set_cur_epoch(self, epoch):
        self.cur_epoch = epoch

    def step(self):
        self.cur_epoch += 1

    def to(self, device):
        super().to(device)
        self.bkg = self.bkg.to(device)
        self.ground = self.ground.to(device)
        self.blocks = self.blocks.to(device)
        self.renderer = self.renderer.to(device)
        self.renderer_fine = self.renderer_fine.to(device)
        self.renderer_env = self.renderer_env.to(device)
        self.renderer_light = self.renderer_light.to(device)
        return self

    @property
    def bkg_n_faces(self):
        return self.bkg.num_faces_per_mesh().sum().item()

    @property
    def ground_n_faces(self):
        return self.ground.num_faces_per_mesh().sum().item()

    @property
    def env_n_faces(self):
        return self.bkg_n_faces + self.ground_n_faces

    @property
    def blocks_n_faces(self):
        return self.blocks.num_faces_per_mesh().sum().item()

    def forward(self, inp, labels):
        rec = self.predict(inp, labels)
        return self.compute_losses(inp['imgs'], rec)

    def predict(self, inp, labels, w_edges=False, filter_transparent=False):
        B, gt, R_tgt, T_tgt = len(inp['imgs']), inp['imgs'], inp['R'], inp['T']
        if 'K' in inp and self.renderer.cameras.K is None:
            self.renderer.update_cameras(device=gt.device, K=inp['K'][0:1])
            self.renderer_fine.update_cameras(device=gt.device, K=inp['K'][0:1])
            self.renderer_env.update_cameras(device=gt.device, K=inp['K'][0:1])
            self.renderer_light.update_cameras(device=gt.device, K=inp['K'][0:1])

        fine_learning = not self.is_live('coarse_learning')
        filter_tsp = filter_transparent or fine_learning
        renderer = self.renderer_fine if fine_learning else self.renderer
        if self.decouple_rendering:
            env = join_meshes_as_scene([self.build_bkg(world_coord=True), self.build_ground(world_coord=True)])
            rec_env = self.renderer_env(env.extend(B), R=R_tgt, T=T_tgt).split([3, 1], dim=1)[0]
            blocks = self.build_blocks(filter_transparent=filter_tsp, as_scene=True)
            if len(blocks) > 0:
                # if filter_transparent, no need to render with smooth opacities
                alpha = None if filter_tsp else self._alpha.repeat_interleave(self.BNF).repeat(B)
                rec_fg, mask = renderer(blocks.extend(B), R=R_tgt, T=T_tgt, faces_alpha=alpha).split([3, 1], dim=1)
            else:
                rec_fg, mask = torch.zeros_like(rec_env), torch.zeros_like(rec_env)
            rec = rec_fg * mask + (1 - mask) * rec_env

        else:
            scene = self.build_scene(filter_transparent=filter_tsp)
            if not filter_tsp:
                alpha_env = torch.ones(self.env_n_faces, device=gt.device)
                alpha = torch.cat([alpha_env, self._alpha.repeat_interleave(self.BNF)], dim=0).repeat(B)
            else:
                alpha = None
            rec, mask = renderer(scene.extend(B), R=R_tgt, T=T_tgt, faces_alpha=alpha).split([3, 1], dim=1)

        if w_edges:
            if self.decouple_rendering:
                scene = join_meshes_as_scene([env, blocks]) if len(blocks) > 0 else env
            colors = self.get_scene_face_colors(filter_transparent=filter_tsp).repeat(B, 1)
            rec = renderer.draw_edges(rec, scene.extend(B), R_tgt, T_tgt, colors=colors)
        return rec

    def predict_synthetic(self, inp, labels):
        B, R_tgt, T_tgt = len(inp['imgs']), inp['R'], inp['T']
        blocks = self.build_blocks(filter_transparent=True, synthetic_colors=True, as_scene=True)
        if len(blocks) > 0:
            rec = self.renderer_light(blocks.extend(B), R=R_tgt, T=T_tgt, viz_purpose=True)[:, :3]
        else:
            rec = torch.ones_like(inp['imgs'])
        return rec

    def build_scene(self, filter_transparent=False, w_bkg=True, reduce_ground=False, synthetic_colors=False):
        meshes = []
        if w_bkg:
            meshes.append(self.build_bkg(synthetic_colors=synthetic_colors))
        meshes.append(self.build_ground(reduced=reduce_ground, synthetic_colors=synthetic_colors))
        blocks = self.build_blocks(filter_transparent, synthetic_colors=synthetic_colors)
        if len(blocks) > 0:
            meshes.append(blocks)
        N_meshes = len(meshes) - 1 + len(blocks)
        if N_meshes > 1:
            scene = join_meshes_as_scene(meshes)
        else:
            scene = meshes[0] if len(meshes) > 0 else self.build_bkg(synthetic_colors=synthetic_colors)
        verts, faces = scene.get_mesh_verts_faces(0)
        verts = (verts[None] * self.S_world) @ self.R_world + self.T_world[:, None]
        return Meshes(verts, faces[None], scene.textures)

    def build_bkg(self, reduced=False, world_coord=False, synthetic_colors=False):
        verts, faces = [t[None] for t in self.bkg.get_mesh_verts_faces(0)]
        if reduced:
            verts = verts * 3 / self.z_far
        if world_coord:
            verts = (verts * self.S_world) @ self.R_world + self.T_world[:, None]
        maps = torch.sigmoid(self.texture_bkg) if not synthetic_colors else torch.ones_like(self.texture_bkg)
        self._bkg_maps = maps
        # Regularization
        if self.training and self.is_live('decimate_txt'):
            sub_maps = F.avg_pool2d(maps.permute(0, 3, 1, 2), kernel_size=self.decim_factor, stride=self.decim_factor)
            maps = F.interpolate(sub_maps, scale_factor=self.decim_factor).permute(0, 2, 3, 1)

        return Meshes(verts, faces, textures=TexturesUV(maps, faces, self.bkg_verts_uvs[None], align_corners=True))

    def build_ground(self, reduced=False, world_coord=False, synthetic_colors=False):
        S_ground = 1. if not reduced else torch.Tensor([3 / self.z_far, 1, 3 / self.z_far]).to(self.bkg.device)
        verts, faces = [t[None] for t in self.ground.get_mesh_verts_faces(0)]
        verts = (verts * S_ground) @ rotation_6d_to_matrix(self.R_6d_ground) + self.T_ground[:, None]
        if world_coord:
            verts = (verts * self.S_world) @ self.R_world + self.T_world[:, None]
        maps = torch.sigmoid(self.texture_ground) if not synthetic_colors else torch.ones_like(self.texture_ground)
        self._ground_maps = maps
        # Regularization
        if self.training and self.is_live('decimate_txt'):
            sub_maps = F.avg_pool2d(maps.permute(0, 3, 1, 2), kernel_size=self.decim_factor, stride=self.decim_factor)
            maps = F.interpolate(sub_maps, scale_factor=self.decim_factor).permute(0, 2, 3, 1)

        return Meshes(verts, faces, textures=TexturesUV(maps, faces, self.ground_verts_uvs[None], align_corners=True))

    def build_blocks(self, filter_transparent=False, world_coord=False, as_scene=False, synthetic_colors=False, filter_killed=True):
        coarse_learning = self.training and self.is_live('coarse_learning')
        S, R, T = self.S.exp() + self.scale_min, rotation_6d_to_matrix(self.R_6d), self.T
        if self.opacity_noise and coarse_learning:
            alpha_logit = self.alpha_logit + self.opacity_noise * torch.randn_like(self.alpha_logit)
        else:
            alpha_logit = self.alpha_logit
        self._alpha = torch.sigmoid(alpha_logit)
        self._alpha_full = self._alpha.clone()  # this tensor won't be filtered / altered based on opacities
        maps = torch.sigmoid(self.textures)
        if synthetic_colors:
            values = torch.linspace(0, 1, self.n_blocks + 1)[1:]
            colors = torch.from_numpy(get_fancy_cmap()(values.cpu().numpy())).float().to(maps.device)
            maps = colors[:, None, None].expand(-1, self.txt_size, self.txt_size, -1)
        verts = (self.get_blocks_verts() * S[:, None]) @ R + T[:, None]
        faces = self.blocks.faces_padded()
        self._blocks_maps, self._blocks_SRT = maps, (S, R, T)

        # Filter blocks based on opacities
        if filter_transparent or (self.kill_blocks and filter_killed):
            if filter_transparent:
                mask = torch.sigmoid(self.alpha_logit) > 0.5
            else:
                mask = torch.sigmoid(self.alpha_logit) > 0.01
            self._alpha_full = self._alpha_full * mask
            NB = sum(mask).item()
            if NB == 0:
                verts, faces, maps = [], [], []
            else:
                verts, faces, maps, self._alpha = verts[mask], faces[mask], maps[mask], self._alpha[mask]
        else:
            NB = self.n_blocks

        # Regularization
        if len(maps) > 0 and coarse_learning:
            if self.is_live('decimate_txt'):
                sub_maps = F.avg_pool2d(maps.permute(0, 3, 1, 2), self.decim_factor, stride=self.decim_factor)
                maps = F.interpolate(sub_maps, scale_factor=self.decim_factor).permute(0, 2, 3, 1)

        # Build textures and meshes object
        verts_uvs = self.block_verts_uvs[None].expand(self.n_blocks, -1, -1)[:NB] if NB != 0 else []
        faces_uvs = self.block_faces_uvs[None].expand(self.n_blocks, -1, -1)[:NB] if NB != 0 else []
        if len(maps) > 0:
            p_left, p_right = self.txt_padding
            maps = F.pad(maps.permute(0, 3, 1, 2), pad=(p_left, p_right, 0, 0), mode='circular').permute(0, 2, 3, 1)
        txt = TexturesUV(maps, faces_uvs, verts_uvs, align_corners=True)
        if (world_coord or as_scene) and len(verts) > 0:
            verts = (verts * self.S_world) @ self.R_world + self.T_world[:, None]
        blocks = Meshes(verts, faces, textures=txt)
        return join_meshes_as_scene(blocks) if (as_scene and len(blocks) > 0) else blocks

    def get_blocks_verts(self):
        eps1, eps2 = (self.sq_eps.sigmoid() * 1.8 + 0.1).split([1, 1], dim=-1)
        verts = parametric_sq(self.sq_eta, self.sq_omega, eps1, eps2) * self.ratio_block_scene
        self._blocks_eps = eps1, eps2
        return verts

    def sample_points_from_blocks(self, N_points=500):
        eps1, eps2 = (self.sq_eps.sigmoid() * 1.8 + 0.1).split([1, 1], dim=-1)
        S, R, T = self.S.exp() + self.scale_min, rotation_6d_to_matrix(self.R_6d), self.T
        points = sample_sq(eps1, eps2, scale=S * self.ratio_block_scene, N_points=N_points)  # NP3
        points = points @ R + T[:, None]
        return points

    def compute_losses(self, imgs, rec):
        losses = {k: torch.tensor(0.0, device=imgs.device) for k in self.loss_weights}

        coarse_learning = self.is_live('coarse_learning')
        # Pixel-wise reconstrution error on RGB values
        if 'rgb' in losses:
            losses['rgb'] = self.loss_weights['rgb'] * self.criterion(imgs, rec)
        # Perceptual loss
        if 'perceptual' in losses:
            factor = 1 if coarse_learning else 0.1
            losses['perceptual'] = self.loss_weights['perceptual'] * factor * self.perceptual_loss(imgs, rec)
        # Parsimony
        if 'parsimony' in losses:
            factor = 1 if coarse_learning else 0
            alpha = self._alpha_full if coarse_learning else (self._alpha_full > 0.5).float()
            losses['parsimony'] = self.loss_weights['parsimony'] * factor * safe_pow(alpha, 0.5).mean()
        # TV loss
        if 'tv' in losses:
            factor = 1 if coarse_learning else 0.1
            tv_loss = sum([self.tv_norm(torch.diff(self._bkg_maps, dim=k)).mean() for k in [1, 2]])
            if len(self._blocks_maps) > 0:
                # we use mapping continuity in TV
                dx = self.tv_norm(torch.diff(self._blocks_maps, dim=2, append=self._blocks_maps[:, :, 0:1]))
                dy = self.tv_norm(torch.diff(self._blocks_maps, dim=1))
                tv_loss += (dx.sum(0).mean() + dy.sum(0).mean())  # sum over blocks so that each map receives same grad
            tv_loss += sum([self.tv_norm(torch.diff(self._ground_maps, dim=k)).mean() for k in [1, 2]]) * factor
            losses['tv'] = self.loss_weights['tv'] * factor * tv_loss
        # Overlap
        if 'overlap' in losses:
            factor = 1 if coarse_learning else 0
            N = self.n_blocks
            with torch.no_grad():
                points = torch.rand(N, OVERLAP_N_POINTS, 3, device=rec.device) * 2 - 1
                S, R, T = self._blocks_SRT
                points = (points * self.ratio_block_scene * S[:, None]) @ R + T[:, None]
                points = points.view(-1, 3)[None].expand(N, -1, -1)

            eps1, eps2 = self._blocks_eps
            points_inv = ((points - T[:, None]) @ R.transpose(1, 2)) / (S[:, None] * self.ratio_block_scene)
            sdf = implicit_sq(points_inv, eps1, eps2, as_sdf=2)
            occupancy = torch.sigmoid(-sdf / OVERLAP_TEMPERATURE)
            alpha = self._alpha_full if coarse_learning else (self._alpha_full > 0.5).float()
            occupancy = occupancy * alpha[:, None]
            overlap_loss = (occupancy.sum(0) - OVERLAP_N_BLOCKS).clamp(0).mean()
            losses['overlap'] = self.loss_weights['overlap'] * factor * overlap_loss

        losses['total'] = sum(losses.values())
        return losses

    def get_opacities(self):
        alpha = torch.sigmoid(self.alpha_logit)
        if self.kill_blocks:
            alpha = alpha * (alpha > 0.01)
        return alpha

    @torch.no_grad()
    def get_nb_opaque_blocks(self):
        return (self.get_opacities() > 0.5).sum().item()

    @torch.no_grad()
    def get_scene_face_colors(self, filter_transparent=False, w_env=True):
        val_blocks = torch.linspace(0, 1, self.n_blocks + 1)[1:]
        if filter_transparent:
            val_blocks = val_blocks[self.get_opacities().cpu() > 0.5]
        elif self.kill_blocks:
            val_blocks = val_blocks[self.get_opacities().cpu() > 0.01]
        cmap = get_fancy_cmap()
        NFE = self.env_n_faces if w_env else 0
        values = torch.cat([torch.zeros(NFE) , val_blocks.repeat_interleave(self.BNF)])
        colors = cmap(values.numpy())
        return torch.from_numpy(colors).float().to(self.bkg.device)

    @torch.no_grad()
    def get_arranged_block_txt(self):
        maps = torch.sigmoid(self.textures).permute(0, 3, 1, 2)
        ncol, nrow = 5, len(maps) // 5
        rows = [torch.cat([maps[k] for k in range(ncol*i, ncol*(i+1))], dim=2) for i in range(nrow)]
        return torch.cat(rows, dim=1)[None]

    @torch.no_grad()
    def load_state_dict(self, state_dict):
        unloaded_params = []
        state = self.state_dict()
        for name, param in safe_model_state_dict(state_dict).items():
            name = name.replace('spq_', 'sq_')  # Backward compatibility
            if name in state:
                try:
                    state[name].copy_(param.data if isinstance(param, nn.Parameter) else param)
                except RuntimeError:
                    state[name].copy_(param.data if isinstance(param, nn.Parameter) else param)
                    print_warning(f'Error load_state_dict param={name}: {list(param.shape)}, {list(state[name].shape)}')
            else:
                unloaded_params.append(name)
        if len(unloaded_params) > 0:
            print_warning(f'load_state_dict: {unloaded_params} not found')

    def is_live(self, name):
        milestone = getattr(self, name)
        if isinstance(milestone, bool):
            return milestone
        else:
            return True if self.cur_epoch < milestone else False

    @torch.no_grad()
    def quantitative_eval(self, loader, device, hard_inference=True):
        self.eval()
        opacities = self.get_opacities()
        n_blocks = (opacities > 0.5).sum().item()

        mse_func = get_loss('mse')().to(device)
        ssim_func = get_loss('ssim')(padding=False).to(device)
        lpips_func = get_loss('lpips')().to(device)
        loss_tot, loss_rec, psnr, ssim, lpips = [AverageMeter() for _ in range(5)]
        scene = self.build_scene(filter_transparent=True)
        for j, (inp, labels) in enumerate(loader):
            inp = torch_to(inp, device)
            imgs, N = inp['imgs'], len(inp['imgs'])
            if hard_inference:
                rec = self.renderer(scene.extend(N), inp['R'], inp['T'], viz_purpose=True)[:, :3]
            else:
                rec = self.predict(inp, labels, filter_transparent=True)
            losses = self.compute_losses(imgs, rec)
            loss_tot.update(losses['total'], N=N)
            loss_rec.update(sum([losses.get(name, 0.) for name in ['rgb', 'perceptual']]), N=N)
            psnr.update(mse2psnr(mse_func(imgs, rec)), N=N)
            ssim.update(1 - ssim_func(imgs, rec).mean(), N=N)
            lpips.update(lpips_func(imgs, rec), N=N)

        return OrderedDict(
            [('n_blocks', n_blocks), ('L_tot', loss_tot.avg), ('L_rec', loss_rec.avg),
             ('PSNR', psnr.avg), ('SSIM', ssim.avg), ('LPIPS', lpips.avg)]
            + [(f'alpha{k}', alpha.item()) for k, alpha in enumerate(opacities)]
        )

    @torch.no_grad()
    def qualitative_eval(self, loader, device, path=None, NV=240):
        path = path or Path('.')
        self.eval()

        # Textures
        out = path_mkdir(path / 'textures')
        convert_to_img(torch.sigmoid(self.texture_bkg).permute(0, 3, 1, 2)).save(out / 'bkg.png')
        convert_to_img(torch.sigmoid(self.texture_ground).permute(0, 3, 1, 2)).save(out / 'ground.png')
        for k, img in enumerate(torch.sigmoid(self.textures).permute(0, 3, 1, 2)):
            convert_to_img(img).save(out / f'block_{str(k).zfill(2)}.png')

        # Basic 3D
        meshes = self.build_scene(filter_transparent=True)
        # colors = self.get_scene_face_colors(filter_transparent=True)
        colors = self.get_scene_face_colors(filter_transparent=True, w_env=False)
        save_mesh_as_video(meshes, path / 'rotated_mesh.mp4', renderer=self.renderer)
        save_mesh_as_obj(meshes, path / 'mesh_full.obj')
        clean_mesh = self.build_scene(filter_transparent=True, w_bkg=False, reduce_ground=True)
        save_mesh_as_obj(clean_mesh, path / 'mesh.obj')
        syn_blocks = self.build_blocks(filter_transparent=True, synthetic_colors=True, as_scene=True)
        if len(syn_blocks) == 0:
            return None
        # GT pointcloud
        gt = loader.dataset.pc_gt
        with use_seed(123):
            gt = gt[torch.randperm(len(gt))[:3000]]
        save_ply(path / 'gt.ply', gt)

        # Create renderers
        renderer, renderer_light = self.renderer, self.renderer_light

        # Input specific
        count, N = 0, 10
        R_traj, T_traj = [t.to(device) for t in get_circle_traj(N_views=NV)]
        n_zeros = int(np.log10(N - 1)) + 1
        BS = loader.batch_size
        for j, (inp, labels) in enumerate(loader):
            if count >= N:
                break
            inp = torch_to(inp, device)
            img_src, R_src, T_src = inp['imgs'], inp['R'], inp['T']
            B = min(len(img_src), N - count)
            for k in range(B):
                i = str(j*BS+k).zfill(n_zeros)
                img = img_src[k]
                convert_to_img(img).save(path / f'{i}_inp.png')
                R, T = R_src[k:k+1], T_src[k:k+1]
                rec = renderer(meshes, R, T, viz_purpose=True)[:, :3]
                convert_to_img(rec).save(path / f'{i}_rec.png')
                convert_to_img(renderer.draw_edges(rec, syn_blocks, R, T, colors)).save(path / f'{i}_rec_col.png')
                convert_to_img(renderer.draw_edges(img, syn_blocks, R, T, colors)).save(path / f'{i}_rec_col_inp.png')
                rec = renderer_light(syn_blocks, R, T, viz_purpose=True)[:, :3]
                convert_to_img(rec).save(path / f'{i}_rec_syn_nobkg.png')
                rec_wedges = renderer_light.draw_edges(rec, syn_blocks, R, T, linewidth=0.7, colors=(0.3, 0.3, 0.3))
                convert_to_img(rec_wedges).save(path / f'{i}_rec_syn_nobkg_edged.png')
                R, T = R @ R_traj, T.expand(NV, -1)
                save_trajectory_as_video(meshes, path / f'{i}_rec_traj.mp4', R=R, T=T, renderer=renderer)
                save_trajectory_as_video(syn_blocks, path / f'{i}_rec_traj_syn.mp4', R=R, T=T, renderer=renderer_light)
            count += B

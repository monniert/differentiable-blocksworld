from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from pytorch3d.renderer import (FoVPerspectiveCameras, RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
                                PointLights, DirectionalLights, Materials, look_at_view_transform,
                                PerspectiveCameras, AmbientLights)
from pytorch3d.renderer.mesh.shader import SoftPhongShader
from pytorch3d.renderer.mesh.shading import phong_shading, flat_shading, gouraud_shading

from utils.image import save_gif, convert_to_img, save_video
from utils.pytorch import get_torch_device


LAYERED_SHADER = True
SHADING_TYPE = 'raw'
VIZ_IMG_SIZE = 256
EPS = 1e-8
DIRECTION_LIGHT = [1, 0.25, -1]


class Renderer(nn.Module):
    def __init__(self, img_size, **kwargs):
        super().__init__()
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self._init_kwargs = deepcopy(kwargs)
        self.init_cameras(**kwargs.pop('cameras', {}))
        self.init_lights(**kwargs.pop('lights', {}))
        blend_kwargs = {'sigma': kwargs.pop('sigma', 1e-4),
                        'background_color': kwargs.pop('background_color', (0, 0, 0))}
        n_faces = kwargs.pop('faces_per_pixel', 25)
        p_correct = kwargs.pop('perspective_correct', None)
        z_clip = kwargs.pop('z_clip', None)
        blend_params = BlendParams(**blend_kwargs)
        s_kwargs = {'cameras': self.cameras, 'lights': self.lights, 'blend_params': blend_params,
                    'debug': kwargs.pop('debug', False)}
        if kwargs.pop('layered_shader', LAYERED_SHADER):
            shader_cls = LayeredShader
            s_kwargs['clip_inside'] = kwargs.pop('clip_inside', True)
            s_kwargs['shading_type'] = kwargs.pop('shading_type', SHADING_TYPE)
            s_kwargs['detach_bary'] = kwargs.pop('detach_bary', False)
        else:
            shader_cls = SoftPhongShaderPlus
        self.r_kwargs = {'perspective_correct': p_correct, 'clip_barycentric_coords': True, 'z_clip_value': z_clip}
        assert len(kwargs) == 0, kwargs

        # approximative differentiable renderer for training
        raster_settings = RasterizationSettings(image_size=self.img_size,
                                                blur_radius=np.log(1./1e-4-1.)*blend_params.sigma,
                                                faces_per_pixel=n_faces, **self.r_kwargs)
        self.renderer = MeshRenderer(MeshRasterizer(cameras=self.cameras, raster_settings=raster_settings),
                                     shader_cls(**s_kwargs))

        # exact anti-aliased rendering for visualization
        s_kwargs['blend_params'] = BlendParams(background_color=blend_kwargs['background_color'], sigma=0)
        raster_settings = RasterizationSettings(image_size=(self.img_size[0]*4, self.img_size[1]*4), **self.r_kwargs)
        self.viz_renderer = VizMeshRenderer(MeshRasterizer(cameras=self.cameras, raster_settings=raster_settings),
                                            shader_cls(**s_kwargs))

    def init_cameras(self, **kwargs):
        kwargs = deepcopy(kwargs)
        name = kwargs.pop('name', 'fov')
        cam_cls = {'fov': FoVPerspectiveCameras, 'perspective': PerspectiveCameras}[name]
        self.cam_kwargs = kwargs
        self.cameras = cam_cls(**kwargs)

    def init_lights(self, **kwargs):
        kwargs = deepcopy(kwargs)
        name = kwargs.pop('name', 'ambient')
        light_cls = {'ambient': AmbientLights, 'directional': DirectionalLights, 'point': PointLights}[name]
        self.lights = light_cls(**kwargs)
        if name == 'directional':
            self.lights._direction = self.lights.direction
            self.lights._ambient_color = self.lights.ambient_color
            self.lights._diffuse_color = self.lights.diffuse_color
            self.lights._specular_color = self.lights.specular_color

    @property
    def init_kwargs(self):
        return deepcopy(self._init_kwargs)

    def forward(self, meshes, R, T, viz_purpose=False, **kwargs):
        # XXX bug when using perspective_correct, see https://github.com/facebookresearch/pytorch3d/issues/561
        # setting eps fixes the issue
        if isinstance(self.lights, DirectionalLights):
            direction = self.lights._direction
            self.update_lights(direction @ R.transpose(1, 2))

        if viz_purpose:
            res = self.viz_renderer(meshes, R=R, T=T, eps=EPS, **kwargs)
        else:
            res = self.renderer(meshes, R=R, T=T, eps=EPS, **kwargs)

        if isinstance(self.lights, DirectionalLights):
            self.update_lights(direction)
        return res  # BCHW

    def to(self, device):
        super().to(device)
        self.renderer = self.renderer.to(device)
        self.viz_renderer = self.viz_renderer.to(device)
        return self

    def get_copy_cameras(self, **kwargs):
        merged_kwargs = deepcopy(self.cam_kwargs)
        merged_kwargs.update(kwargs)
        return self.cameras.__class__(**merged_kwargs)

    def update_cameras(self, **kwargs):
        self.cameras = self.get_copy_cameras(**kwargs)
        self.renderer.rasterizer.cameras = self.cameras
        self.viz_renderer.rasterizer.cameras = self.cameras
        self.renderer.shader.cameras = self.cameras
        self.viz_renderer.shader.cameras = self.cameras

    def update_lights(self, direction=None, ka=None, kd=None, ks=None):
        if direction is not None:
            self.lights.direction = direction
        if ka is not None:
            self.lights.ambient_color = ka
        if kd is not None:
            self.lights.diffuse_color = kd
        if ks is not None:
            self.lights.specular_color = ks

    def reset_default_lights(self):
        self.lights.direction = self.lights._direction
        self.lights.ambient_color = self.lights._ambient_color
        self.lights.diffuse_color = self.lights._diffuse_color
        self.lights.specular_color = self.lights._specular_color

    @torch.no_grad()
    def render_edges(self, meshes, R, T, image_size=None, linewidth=1, return_pix2face=False, faces_per_pixel=1):
        image_size = image_size or self.img_size
        raster_settings = RasterizationSettings(image_size=image_size, faces_per_pixel=faces_per_pixel, **self.r_kwargs)
        fragments = MeshRasterizer(self.cameras, raster_settings)(meshes, R=R, T=T, eps=EPS)
        # dist is the signed squared distance in NDC space i.e. [-1, 1] for the smallest length,
        # the pixel size along the smallest length in NDC is thus 2 / min(image_size)
        mask = (-fragments.dists < (linewidth * 2 / min(image_size)) ** 2).float()[:, None]  # B1HWK
        mask = mask.max(-1)[0]
        if return_pix2face:
            return mask, fragments.pix_to_face[..., 0]  # BHW
        else:
            return mask

    def draw_edges(self, img, meshes, R=None, T=None, colors=None, linewidth=1, antialias=True):
        if R is None:
            R = torch.eye(3, device=meshes.device)[None].expand(len(meshes), -1, -1)
        if T is None:
            T = torch.zeros(1, 3, device=meshes.device).expand(len(meshes), -1)
        if colors is None:
            colors = (1, 0, 0)
        img_size = img.shape[-2:] if isinstance(img, torch.Tensor) else img.size[::-1]
        if antialias:
            img_size = (img_size[0] * 4, img_size[1]*4)
            linewidth = linewidth * 4
        if isinstance(colors, (list, tuple)):
            colors = torch.Tensor(colors)

        mask, pix2face = self.render_edges(meshes, R, T, image_size=img_size, linewidth=linewidth, return_pix2face=True)
        if len(colors.shape) == 2:
            face_img = colors[pix2face].permute(0, 3, 1, 2)  # one color per face
        else:
            face_img = torch.Tensor(colors)[:, None, None].to(mask.device).expand(-1, *mask.shape[2:])
        if antialias:
            mask, face_img = [F.avg_pool2d(t, kernel_size=4, stride=4) for t in [mask, face_img]]

        if isinstance(img, torch.Tensor):
            img = img * (1 - mask) + mask * face_img
        else:
            img = img.copy()
            img.paste(convert_to_img(face_img), mask=convert_to_img(mask).convert('L'))
        return img


class VizMeshRenderer(MeshRenderer):
    """Renderer for visualization, with anti-aliasing"""
    @torch.no_grad()
    def __call__(self, *input, **kwargs):
        res = super().__call__(*input, **kwargs)
        return F.avg_pool2d(res, kernel_size=4, stride=4)


class LayeredShader(nn.Module):
    def __init__(self, device='cpu', cameras=None, lights=None, materials=None, blend_params=None, clip_inside=True,
                 shading_type='phong', detach_bary=False, debug=False):
        super().__init__()
        self.lights = lights if lights is not None else DirectionalLights(device=device)
        self.materials = (materials if materials is not None else Materials(device=device))
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()
        self.clip_inside = clip_inside
        if shading_type == 'phong':
            shading_fn = phong_shading
        elif shading_type == 'flat':
            shading_fn = flat_shading
        elif shading_type == 'gouraud':
            shading_fn = gouraud_shading
        elif shading_type == 'raw':
            shading_fn = lambda x: x
        else:
            raise NotImplementedError
        self.shading_fn = shading_fn
        self.shading_type = shading_type
        self.detach_bary = detach_bary
        self.debug = debug

    def to(self, device):
        # Manually move to device modules which are not subclasses of nn.Module
        cameras = self.cameras
        if cameras is not None:
            self.cameras = cameras.to(device)
        self.materials = self.materials.to(device)
        self.lights = self.lights.to(device)
        return self

    def forward(self, fragments, meshes, **kwargs):
        blend_params = kwargs.get("blend_params", self.blend_params)
        faces_alpha = kwargs.get('faces_alpha')
        if self.detach_bary:
            fragments.bary_coords.detach_()

        if self.shading_type == 'raw':
            colors = meshes.sample_textures(fragments)
            if not torch.all(self.lights.ambient_color == 1):
                colors *= self.lights.ambient_color
        else:
            sh_kwargs = {'meshes': meshes, 'fragments': fragments, 'cameras': kwargs.get("cameras", self.cameras),
                         'lights': kwargs.get("lights", self.lights),
                         'materials': kwargs.get("materials", self.materials)}
            if self.shading_type != 'gouraud':
                sh_kwargs['texels'] = meshes.sample_textures(fragments)
            colors = self.shading_fn(**sh_kwargs)
        rec = layered_rgb_blend(colors, fragments, blend_params, clip_inside=self.clip_inside, debug=self.debug,
                                faces_alpha=faces_alpha)
        return rec


def layered_rgb_blend(colors, fragments, blend_params, clip_inside=True, debug=False, faces_alpha=None):
    N, H, W, K = fragments.pix_to_face.shape
    device = fragments.pix_to_face.device
    pixel_colors = torch.ones((N, H, W, 4), dtype=colors.dtype, device=colors.device)
    background_ = blend_params.background_color
    if not isinstance(background_, torch.Tensor):
        background = torch.tensor(background_, dtype=torch.float32, device=device)
    else:
        background = background_.to(device)

    mask = fragments.pix_to_face >= 0  # mask for padded pixels.
    if blend_params.sigma == 0:
        alpha = (fragments.dists <= 0).float() * mask
    elif clip_inside:
        alpha = torch.exp(-fragments.dists.clamp(0) / blend_params.sigma) * mask
    else:
        alpha = torch.sigmoid(-fragments.dists / blend_params.sigma) * mask
    if faces_alpha is not None:
        pix2face_clean = fragments.pix_to_face.view(-1).clamp(0)
        alpha = alpha * faces_alpha.gather(0, pix2face_clean).view(alpha.shape)
        # alpha = alpha * faces_alpha[fragments.pix_to_face]  # XXX much much slower
    occ_alpha = torch.cumprod(1.0 - alpha, dim=-1)
    occ_alpha = torch.cat([torch.ones(N, H, W, 1, device=device), occ_alpha], dim=-1)
    colors = torch.cat([colors, background[None, None, None, None].expand(N, H, W, 1, -1)], dim=-2)
    alpha = torch.cat([alpha, torch.ones(N, H, W, 1, device=device)], dim=-1)
    pixel_colors[..., :3] = (occ_alpha[..., None] * alpha[..., None] * colors).sum(-2)
    pixel_colors[..., 3] = 1 - occ_alpha[:, :, :, -1]
    pixel_colors = pixel_colors.permute(0, 3, 1, 2)

    if debug:
        return colors, alpha, occ_alpha, pixel_colors
    else:
        return pixel_colors  # BCHW


class SoftPhongShaderPlus(SoftPhongShader):
    """Rewriting to permute output tensor + working `to` method for multi-gpus"""
    def forward(self, fragments, meshes, **kwargs):
        return super().forward(fragments, meshes, **kwargs).permute(0, 3, 1, 2)

    def to(self, device):
        cameras = self.cameras
        if cameras is not None:
            self.cameras = cameras.to(device)
        self.materials = self.materials.to(device)
        self.lights = self.lights.to(device)
        return self


@torch.no_grad()
def render_rotated_views(mesh, n_views=50, elev=30, dist=2.5, R=None, T=None, bkg=None,
                         renderer=None, rend_kwargs=None, eye_light=False, device=None):
    img_size = VIZ_IMG_SIZE if renderer is None else renderer.img_size
    device = get_torch_device() if device is None else device
    rend_kwargs = {} if rend_kwargs is None else rend_kwargs
    if renderer is not None:
        rend_kwargs = renderer.init_kwargs
        if hasattr(renderer.cameras, 'K'):
            rend_kwargs['cameras']['K'] = renderer.cameras.K
    renderer = Renderer(img_size, **rend_kwargs)
    if eye_light:
        if R is not None:
            raise NotImplementedError
        if isinstance(renderer.lights, AmbientLights):
            kwargs = renderer.init_kwargs
            kwargs['lights'] = {'name': 'directional', 'direction': [[0, 0, -1]], 'ambient_color': [[0.6, 0.6, 0.6]],
                                'diffuse_color': [[0.4, 0.4, 0.4]], 'specular_color': [[0., 0., 0.]]}
            kwargs['shading_type'] = 'phong'
            kwargs['faces_per_pixel'] = 1
            renderer = Renderer(img_size, **kwargs)
    elev, dist = 0 if R is not None else elev, 0 if T is not None else dist
    R, T = R if R is not None else torch.eye(3).to(device), T if T is not None else torch.zeros(3).to(device)

    if bkg is not None:
        if bkg.shape[-1] != img_size:
            bkg = F.interpolate(bkg[None], size=(img_size, img_size), mode='bilinear', align_corners=False)[0]
    mesh, renderer = mesh.to(device), renderer.to(device)

    azim = torch.linspace(-180, 180, n_views)
    views, B = [], 10
    for k in range((n_views - 1) // B + 1):
        # we render by batch of B views to avoid OOM
        R_view = look_at_view_transform(dist=1, elev=elev, azim=azim[k*B: (k+1)*B], device=device)[0]
        T_view = torch.Tensor([[0., 0., dist]]).to(device).expand(len(R_view), -1)
        views.append(renderer(mesh.extend(len(R_view)), R_view @ R, T_view + T, viz_purpose=True).clamp(0, 1).cpu())

    rec, alpha = torch.cat(views, dim=0).split([3, 1], dim=1)
    if bkg is not None:
        rec = rec * alpha + (1 - alpha) * bkg.cpu()
    return rec


@torch.no_grad()
def render_views(mesh, R, T, img_size=None, bkg=None, renderer=None, rend_kwargs=None,
                 with_edges=False, linewidth=1, edge_colors=None, eye_light=False, with_alpha=False, device=None):
    img_size = (img_size or VIZ_IMG_SIZE) if renderer is None else renderer.img_size
    img_size = VIZ_IMG_SIZE if renderer is None else renderer.img_size
    device = get_torch_device() if device is None else device
    rend_kwargs = {} if rend_kwargs is None else rend_kwargs
    if renderer is None:
        renderer = Renderer(img_size, **rend_kwargs)
    mesh, renderer = mesh.to(device), renderer.to(device)
    if eye_light:
        if isinstance(renderer.lights, AmbientLights):
            kwargs = renderer.init_kwargs
            kwargs['lights'] = {'name': 'directional', 'direction': [DIRECTION_LIGHT],
                                'ambient_color': [[0.7, 0.7, 0.7]],
                                'diffuse_color': [[0.4, 0.4, 0.4]], 'specular_color': [[0., 0., 0.]]}
            kwargs['shading_type'] = 'phong'
            kwargs['faces_per_pixel'] = 1
            renderer = Renderer(img_size, **kwargs).to(device)

    if bkg is not None:
        if tuple(bkg.shape[-2:]) != renderer.img_size:
            bkg = F.interpolate(bkg[None], size=(img_size, img_size), mode='bilinear', align_corners=False)[0]

    n_views = len(R)
    views, B = [], 10
    for k in range((n_views - 1) // B + 1):
        # we render by batch of B views to avoid OOM
        R_view = R[k*B: (k+1)*B]
        T_view = T[k*B: (k+1)*B]
        B_eff = len(R_view)
        meshes = mesh.extend(B_eff)
        res = renderer(meshes, R_view, T_view, viz_purpose=True)
        if with_edges:
            res, alpha = res.split([3, 1], dim=1)
            if edge_colors is not None and isinstance(edge_colors, torch.Tensor):
                edge_colors = edge_colors.repeat(B_eff, 1)
            res = renderer.draw_edges(res, meshes, R=R_view, T=T_view, linewidth=linewidth, colors=edge_colors)
            res = torch.cat([res, alpha], dim=1)
        views.append(res.cpu())

    if with_alpha:
        return torch.cat(views, dim=0)

    rec, alpha = torch.cat(views, dim=0).split([3, 1], dim=1)
    if bkg is not None:
        rec = rec * alpha + (1 - alpha) * bkg.cpu()
    return rec


def save_mesh_as_gif(mesh, filename, n_views=50, elev=30, dist=2.732, R=None, T=None, bkg=None,
                     renderer=None, rend_kwargs=None, eye_light=False):
    imgs = render_rotated_views(mesh, n_views, elev, dist, R=R, T=T, bkg=bkg, renderer=renderer,
                                rend_kwargs=rend_kwargs, eye_light=eye_light)
    save_gif(imgs, filename)


def save_mesh_as_video(mesh, filename, n_views=240, elev=30, dist=2.732, R=None, T=None, bkg=None,
                       renderer=None, rend_kwargs=None, eye_light=False):
    imgs = render_rotated_views(mesh, n_views, elev, dist, R=R, T=T, bkg=bkg, renderer=renderer,
                                rend_kwargs=rend_kwargs, eye_light=eye_light)
    save_video(imgs, filename)


def save_trajectory_as_gif(mesh, filename, R, T, bkg=None, renderer=None, rend_kwargs=None,
                           with_edges=False, linewidth=1, edge_colors=None, eye_light=False):
    imgs = render_views(mesh, R, T, bkg=bkg, renderer=renderer, rend_kwargs=rend_kwargs, with_edges=with_edges,
                        linewidth=linewidth, edge_colors=edge_colors, eye_light=eye_light)
    save_gif(imgs, filename)


def save_trajectory_as_video(mesh, filename, R, T, bkg=None, renderer=None, rend_kwargs=None,
                             with_edges=False, linewidth=1, edge_colors=None, eye_light=False):
    imgs = render_views(mesh, R, T, bkg=bkg, renderer=renderer, rend_kwargs=rend_kwargs, with_edges=with_edges,
                        linewidth=linewidth, edge_colors=edge_colors, eye_light=eye_light)
    save_video(imgs, filename)


def get_circle_traj(dist=1, a_scale=15, e_scale=15, N_views=50):
    azim = torch.cos(torch.linspace(0, 2, N_views + 1) * np.pi)[:-1] * a_scale - 180
    elev = torch.sin(torch.linspace(0, 2, N_views + 1) * np.pi)[:-1] * e_scale
    return look_at_view_transform(dist, azim=azim, elev=elev)

from typing import Optional
import warnings

from iopath.common.file_io import PathManager
from pathlib import Path
from PIL import Image
import numpy as np
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.io.utils import _open_file
from pytorch3d.loss import mesh_normal_consistency
from pytorch3d.ops import sample_points_from_meshes as sample_points
from pytorch3d.renderer import TexturesVertex, TexturesUV
from pytorch3d.structures import Meshes
from pytorch3d.structures.utils import packed_to_list
from pytorch3d.utils import ico_sphere
import trimesh
from trimesh.voxel.creation import voxelize as trim_voxelize
import torch

from .path import PRIMITIVES_PATH

EPS = 1e-6


def normalize(meshes, center=True, scale_mode='unit_cube', inplace=False, use_center_mass=False):
    if center:
        if use_center_mass:
            offsets = sample_points(meshes, 100000).mean(1)
        else:
            offsets = 0.5 * (meshes.verts_padded().max(1)[0] + meshes.verts_padded().min(1)[0])
        # meshes.offset_vert requires tensor of size (all_V, 3), while offsets is (B, 3)
        NVs = meshes.num_verts_per_mesh()
        offsets = torch.cat([offset[None].expand(nv, -1) for offset, nv in zip(offsets, NVs)], dim=0)
        meshes = meshes.offset_verts_(-offsets) if inplace else meshes.offset_verts(-offsets)

    if scale_mode == 'none' or scale_mode is None:
        scales = 1.
    elif scale_mode == 'unit_cube':
        scales = meshes.verts_padded().abs().flatten(1).max(1)[0] * 2  # [-0.5, 0.5]^3
    elif scale_mode == 'unit_sphere':
        scales = meshes.verts_padded().norm(dim=2).max(1)[0] * 2
    else:
        raise NotImplementedError
    return meshes.scale_verts_(1 / scales) if inplace else meshes.scale_verts(1 / scales)


def repeat(mesh, N):
    """
    Returns N copies using the PyTorch `repeat` convention, compared to the current PyTorch3D function `extend` which
    follows the `repeat_interleave` convention
    """
    assert N >= 1
    if N == 1:
        return mesh

    new_verts_list, new_faces_list = [], []
    for _ in range(N):
        new_verts_list.extend(verts.clone() for verts in mesh.verts_list())
        new_faces_list.extend(faces.clone() for faces in mesh.faces_list())

    textures = mesh.textures
    if isinstance(textures, TexturesVertex):
        new_verts_rgb = textures.verts_features_padded().repeat(N, 1, 1)
        new_textures = TexturesVertex(verts_features=new_verts_rgb)
        new_textures._num_verts_per_mesh = textures._num_verts_per_mesh * N
    elif isinstance(textures, TexturesUV):
        maps = textures.maps_padded().repeat(N, 1, 1, 1)
        uvs = textures.verts_uvs_padded().repeat(N, 1, 1)
        faces = textures.faces_uvs_padded().repeat(N, 1, 1)
        new_textures = TexturesUV(maps, faces, uvs)
        new_textures._num_faces_per_mesh = textures._num_faces_per_mesh * N
    else:
        raise NotImplementedError

    return Meshes(verts=new_verts_list, faces=new_faces_list, textures=new_textures)


def point_to_uv_sphericalmap(X, eps=1e-7, normalize=True):
    """Resulting UV in [0, 1]"""
    if normalize:
        radius = torch.norm(X, dim=-1).clamp(min=eps)
        y = (X[..., 1] / radius).clamp(-1 + eps, 1 - eps)
    else:
        y = X[..., 1]
    theta = torch.acos(-y)    # Inclination: Angle with -Y [0,pi]
    phi = torch.atan2(X[..., 0], X[..., 2])  # Azimuth: Angle with +Z [-pi,pi]
    vv = (theta / np.pi)
    uu = ((phi + np.pi) / (2*np.pi))
    return torch.stack([uu, vv], dim=-1)


def uv_to_point_sphericalmap(uv, radius=1, half_sphere=False):
    """input UV in [0, 1]"""
    phi = np.pi * (uv[..., 0] * 2 - 1)
    theta = np.pi * uv[..., 1]
    if half_sphere:
        theta = theta / 2
    z = torch.sin(theta) * torch.cos(phi)
    x = torch.sin(theta) * torch.sin(phi)
    y = -torch.cos(theta)
    return torch.stack([x, y, z], dim=-1) * radius


def get_icosphere(level=3, order_verts_by=None, colored=False, flip_faces=False):
    mesh = ico_sphere(level)
    if order_verts_by is not None:
        assert isinstance(order_verts_by, int)
        verts, faces = mesh.get_mesh_verts_faces(0)
        N = len(verts)
        indices = sorted(range(N), key=lambda i: verts[i][order_verts_by])
        mapping = torch.zeros(N, dtype=torch.long)
        mapping[indices] = torch.arange(N)
        verts.copy_(verts[indices]), faces.copy_(mapping[faces])

    if flip_faces:
        verts, faces = mesh.get_mesh_verts_faces(0)
        faces = torch.stack([faces[:, 2], faces[:, 1], faces[:, 0]], dim=-1)
        mesh = Meshes(verts[None], faces[None])

    if colored:
        verts = mesh.verts_packed()
        colors = (verts - verts.min(0)[0]) / (verts.max(0)[0] - verts.min(0)[0])
        mesh.textures = TexturesVertex(verts_features=colors[None])
    return mesh


def get_icosphere_uvs(level=3, fix_continuity=False, fix_poles=False, eps=1e-8):
    verts, faces = get_icosphere(level).get_mesh_verts_faces(0)
    verts_uvs = point_to_uv_sphericalmap(verts)
    if fix_continuity:
        # XXX some faces span over the entire uv to connect vertices at both ends, hence yielding wrong mapping
        # we need to redefine those faces with new verts on the same end
        vf_uvs = verts_uvs[faces]
        f_to_process = torch.diff(vf_uvs[..., 0], dim=1, append=vf_uvs[..., 0:1, 0]).abs().max(1)[0] > 0.5
        bad_vf_uvs = vf_uvs[f_to_process]
        vf_u = bad_vf_uvs[..., 0] - 0.5 + eps  # we had an eps to avoid verts == 0 when computing sign
        common_side = torch.sign(vf_u).sum(1)  # we compute where 2 out of the 3 verts are
        mask = torch.sign(vf_u) != common_side[:, None]  # the mask corresponds to the vert not on the same side
        new_vf_u = bad_vf_uvs[..., 0] + common_side[:, None] * mask  # we move the vert by +- 1

        V = len(verts_uvs)
        fixed_verts_uvs = torch.stack([new_vf_u[mask], bad_vf_uvs[..., 1][mask]], dim=-1)
        new_verts_uvs = torch.cat([verts_uvs, fixed_verts_uvs], dim=0)
        new_faces = faces.clone()
        fixed_faces = faces[f_to_process].clone()
        fixed_faces[mask] = (V + torch.arange(0, len(mask)))
        new_faces[f_to_process] = fixed_faces
        faces, verts_uvs = new_faces, new_verts_uvs

    if fix_poles:
        # XXX faces at poles are continuous thus creating strong distortions
        # we here cut the faces and create new vertices in the middle to avoid distortion
        vf_uvs = verts_uvs[faces]
        f_to_process = torch.logical_or(vf_uvs[..., 1].max(1)[0] > 0.99, vf_uvs[..., 1].min(1)[0] < 0.01)
        bad_verts_uvs = vf_uvs[f_to_process]
        verts_v = bad_verts_uvs[..., 1]
        mask = torch.logical_or(verts_v > 0.99, verts_v < 0.01)
        verts_u = ((1 - mask.float()) * bad_verts_uvs[..., 0]).sum(1) / 2

        V = len(verts_uvs)
        fixed_verts_uvs = torch.stack([verts_u, verts_v[mask]], dim=-1)
        new_verts_uvs = torch.cat([verts_uvs, fixed_verts_uvs], dim=0)
        new_faces = faces.clone()
        fixed_faces = faces[f_to_process].clone()
        fixed_faces[mask] = (V + torch.arange(0, len(mask)))
        new_faces[f_to_process] = fixed_faces
        faces, verts_uvs = new_faces, new_verts_uvs

    return faces, verts_uvs


def get_cube():
    return load_objs_as_meshes([PRIMITIVES_PATH / 'cube.obj'], load_textures=False)


def get_cube_uvs():
    faces_uvs = torch.Tensor([
        [1, 3, 0],
        [7, 5, 4],
        [4, 9, 8],
        [11, 2, 10],
        [2, 7, 3],
        [12, 7, 13],
        [1, 2, 3],
        [7, 6, 5],
        [4, 5, 9],
        [11, 6, 2],
        [2, 6, 7],
        [12, 3, 7]
    ]).long()
    verts_uvs = torch.Tensor([
        [0., 3/8],
        [0., 5/8],
        [1/4, 5/8],
        [1/4, 3/8],
        [3/4, 3/8],
        [3/4, 5/8],
        [2/4, 5/8],
        [2/4, 3/8],
        [1., 3/8],
        [1., 5/8],
        [1/4, 7/8],
        [2/4, 7/8],
        [1/4, 1/8],
        [2/4, 1/8],
    ]).float()
    return faces_uvs, verts_uvs


def get_plane():
    return load_objs_as_meshes([PRIMITIVES_PATH / 'plane.obj'], load_textures=False)


def normal_consistency(meshes, icosphere_topology=True, shared_topology=True):
    """Use a x10 faster routine than the one in PyTorch3D when meshes have an icosphere topology"""
    if not icosphere_topology:
        return mesh_normal_consistency(meshes)

    if meshes.isempty():
        return torch.tensor([0.0], dtype=torch.float32, device=meshes.device, requires_grad=True)

    N = len(meshes)
    verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
    faces_packed = meshes.faces_packed()  # (sum(F_n), 3)
    edges_packed = meshes.edges_packed()  # (sum(E_n), 2)
    verts_packed_to_mesh_idx = meshes.verts_packed_to_mesh_idx()  # (sum(V_n),)
    face_to_edge = meshes.faces_packed_to_edges_packed()  # (sum(F_n), 3)
    F = faces_packed.shape[0]  # sum(F_n)

    with torch.no_grad():
        edge_idx = face_to_edge.reshape(F * 3)  # (3 * F,) indexes into edges
        vert_idx = faces_packed.view(1, F, 3).expand(3, F, 3).transpose(0, 1).reshape(3 * F, 3)
        edge_idx, edge_sort_idx = edge_idx.sort()
        vert_idx = vert_idx[edge_sort_idx]
        vert_edge_pair_idx = torch.arange(len(edge_idx), device=meshes.device).view(-1, 2)

    v0_idx = edges_packed[edge_idx, 0]
    v0 = verts_packed[v0_idx]
    v1_idx = edges_packed[edge_idx, 1]
    v1 = verts_packed[v1_idx]

    # two of the following cross products are zeros as they are cross product
    # with either (v1-v0)x(v1-v0) or (v1-v0)x(v0-v0)
    n_temp0 = (v1 - v0).cross(verts_packed[vert_idx[:, 0]] - v0, dim=1)
    n_temp1 = (v1 - v0).cross(verts_packed[vert_idx[:, 1]] - v0, dim=1)
    n_temp2 = (v1 - v0).cross(verts_packed[vert_idx[:, 2]] - v0, dim=1)
    n = n_temp0 + n_temp1 + n_temp2
    n0 = n[vert_edge_pair_idx[:, 0]]
    n1 = -n[vert_edge_pair_idx[:, 1]]
    loss = 1 - torch.cosine_similarity(n0, n1, dim=1)

    if not shared_topology:
        verts_packed_to_mesh_idx = verts_packed_to_mesh_idx[vert_idx[:, 0]]
        verts_packed_to_mesh_idx = verts_packed_to_mesh_idx[vert_edge_pair_idx[:, 0]]
        num_normals = verts_packed_to_mesh_idx.bincount(minlength=N)
        weights = 1.0 / num_normals[verts_packed_to_mesh_idx].float()
        loss = (loss * weights).sum() / N
    else:
        loss = loss.mean()

    return loss


def autopadding_to_3d_cube(grid, size):
    vol = torch.zeros(size, size, size)
    i, j, k = [int((size - grid.size(k)) / 2) for k in range(len(grid.shape))]
    vol[i:i+grid.size(0), j:j+grid.size(1), k:k+grid.size(2)] = grid
    return vol


def voxelize(mesh, size=32):
    verts, faces = mesh.get_mesh_verts_faces(0)
    t_mesh = trimesh.Trimesh(verts.cpu().numpy(), faces.cpu().numpy(), validate=True)
    extent = (verts.max(0)[0] - verts.min(0)[0]).max().item()
    pitch = extent / (size - 1 - EPS)
    grid = trim_voxelize(t_mesh, pitch=pitch, method='subdivide')
    return autopadding_to_3d_cube(torch.Tensor(grid.matrix), size).byte().to(verts.device)


##########################################
# Save mesh as .obj file with support for RGB vertex colors
# inspired by https://github.com/facebookresearch/pytorch3d, see pytorch3d/io/obj_io.py file
##########################################


def save_mesh_as_obj(mesh, filename):
    assert len(mesh) == 1
    verts, faces = mesh.get_mesh_verts_faces(0)
    if isinstance(mesh.textures, TexturesUV):
        txt = mesh.textures
        save_obj(filename, verts, faces, verts_uvs=txt.verts_uvs_padded()[0], faces_uvs=txt.faces_uvs_padded()[0],
                 texture_map=txt.maps_padded()[0])
    elif isinstance(mesh.textures, TexturesVertex):
        verts_rgb = mesh.textures.verts_features_list()[0].clamp(0, 1)
        save_obj(filename, verts, faces, verts_rgb=verts_rgb)
    else:
        save_obj(filename, verts, faces)


def convert_textures_uv_to_vertex(textures_uv, meshes):
    verts_colors_packed = torch.zeros_like(meshes.verts_packed())
    verts_colors_packed[meshes.faces_packed()] = textures_uv.faces_verts_textures_packed()
    return TexturesVertex(packed_to_list(verts_colors_packed, meshes.num_verts_per_mesh()))


def save_obj(
    f,
    verts,
    faces,
    decimal_places: Optional[int] = None,
    path_manager: Optional[PathManager] = None,
    *,
    verts_rgb: Optional[torch.Tensor] = None,
    verts_uvs: Optional[torch.Tensor] = None,
    faces_uvs: Optional[torch.Tensor] = None,
    texture_map: Optional[torch.Tensor] = None,
) -> None:
    """
    Save a mesh to an .obj file.
    Args:
        f: File (str or path) to which the mesh should be written.
        verts: FloatTensor of shape (V, 3) giving vertex coordinates.
        faces: LongTensor of shape (F, 3) giving faces.
        decimal_places: Number of decimal places for saving.
        path_manager: Optional PathManager for interpreting f if
            it is a str.
        verts_uvs: FloatTensor of shape (V, 2) giving the uv coordinate per vertex.
        faces_uvs: LongTensor of shape (F, 3) giving the index into verts_uvs for
            each vertex in the face.
        texture_map: FloatTensor of shape (H, W, 3) representing the texture map
            for the mesh which will be saved as an image. The values are expected
            to be in the range [0, 1],
    """
    if len(verts) and (verts.dim() != 2 or verts.size(1) != 3):
        message = "'verts' should either be empty or of shape (num_verts, 3)."
        raise ValueError(message)

    if len(faces) and (faces.dim() != 2 or faces.size(1) != 3):
        message = "'faces' should either be empty or of shape (num_faces, 3)."
        raise ValueError(message)

    if verts_rgb is not None and (verts.dim() != 2 or verts.size(1) != 3):
        message = "'verts_rgb' should either be empty or of shape (num_verts, 3)."
        raise ValueError(message)

    if faces_uvs is not None and (faces_uvs.dim() != 2 or faces_uvs.size(1) != 3):
        message = "'faces_uvs' should either be empty or of shape (num_faces, 3)."
        raise ValueError(message)

    if verts_uvs is not None and (verts_uvs.dim() != 2 or verts_uvs.size(1) != 2):
        message = "'verts_uvs' should either be empty or of shape (num_verts, 2)."
        raise ValueError(message)

    if texture_map is not None and (texture_map.dim() != 3 or texture_map.size(2) != 3):
        message = "'texture_map' should either be empty or of shape (H, W, 3)."
        raise ValueError(message)

    if path_manager is None:
        path_manager = PathManager()

    save_texture = all([t is not None for t in [faces_uvs, verts_uvs, texture_map]])
    output_path = Path(f)

    # Save the .obj file
    with _open_file(f, path_manager, "w") as f:
        if save_texture:
            # Add the header required for the texture info to be loaded correctly
            obj_header = "\nmtllib {0}.mtl\nusemtl mesh\n\n".format(output_path.stem)
            f.write(obj_header)
        _save(
            f,
            verts,
            faces,
            decimal_places,
            verts_rgb=verts_rgb,
            verts_uvs=verts_uvs,
            faces_uvs=faces_uvs,
            save_texture=save_texture,
        )

    # Save the .mtl and .png files associated with the texture
    if save_texture:
        image_path = output_path.with_suffix(".png")
        mtl_path = output_path.with_suffix(".mtl")
        if isinstance(f, str):
            # Back to str for iopath interpretation.
            image_path = str(image_path)
            mtl_path = str(mtl_path)

        # Save texture map to output folder
        # pyre-fixme[16] # undefined attribute cpu
        texture_map = texture_map.detach().cpu() * 255.0
        image = Image.fromarray(texture_map.numpy().astype(np.uint8))
        with _open_file(image_path, path_manager, "wb") as im_f:
            # pyre-fixme[6] # incompatible parameter type
            image.save(im_f)

        # Create .mtl file with the material name and texture map filename
        # TODO: enable material properties to also be saved.
        with _open_file(mtl_path, path_manager, "w") as f_mtl:
            lines = f"newmtl mesh\n" f"map_Kd {output_path.stem}.png\n"
            f_mtl.write(lines)


def _save(
    f,
    verts,
    faces,
    decimal_places: Optional[int] = None,
    *,
    verts_rgb: Optional[torch.Tensor] = None,
    verts_uvs: Optional[torch.Tensor] = None,
    faces_uvs: Optional[torch.Tensor] = None,
    save_texture: bool = False,
) -> None:

    if len(faces) and (faces.dim() != 2 or faces.size(1) != 3):
        message = "'faces' should either be empty or of shape (num_faces, 3)."
        raise ValueError(message)

    if not (len(verts) or len(faces)):
        warnings.warn("Empty 'verts' and 'faces' arguments provided")
        return

    if verts_rgb is not None and (verts_rgb.dim() != 2 or verts_rgb.size(1) != 3):
        message = "'verts_rgb' should either be None or of shape (num_verts, 3)."
        raise ValueError(message)

    verts, faces = verts.cpu(), faces.cpu()

    lines = ""

    if len(verts):
        if decimal_places is None:
            float_str = "%f"
        else:
            float_str = "%" + ".%df" % decimal_places

        V, D = verts.shape
        for i in range(V):
            vert = [float_str % verts[i, j] for j in range(D)]
            if verts_rgb is not None:
                vert += [float_str % verts_rgb[i, j] for j in range(3)]
            lines += "v %s\n" % " ".join(vert)

    if save_texture:
        if faces_uvs is not None and (faces_uvs.dim() != 2 or faces_uvs.size(1) != 3):
            message = "'faces_uvs' should either be empty or of shape (num_faces, 3)."
            raise ValueError(message)

        if verts_uvs is not None and (verts_uvs.dim() != 2 or verts_uvs.size(1) != 2):
            message = "'verts_uvs' should either be empty or of shape (num_verts, 2)."
            raise ValueError(message)

        # pyre-fixme[16] # undefined attribute cpu
        verts_uvs, faces_uvs = verts_uvs.cpu(), faces_uvs.cpu()

        # Save verts uvs after verts
        if len(verts_uvs):
            uV, uD = verts_uvs.shape
            for i in range(uV):
                uv = [float_str % verts_uvs[i, j] for j in range(uD)]
                lines += "vt %s\n" % " ".join(uv)

    if torch.any(faces >= verts.shape[0]) or torch.any(faces < 0):
        warnings.warn("Faces have invalid indices")

    if len(faces):
        F, P = faces.shape
        for i in range(F):
            if save_texture:
                # Format faces as {verts_idx}/{verts_uvs_idx}
                face = [
                    "%d/%d" % (faces[i, j] + 1, faces_uvs[i, j] + 1) for j in range(P)
                ]
            else:
                face = ["%d" % (faces[i, j] + 1) for j in range(P)]

            if i + 1 < F:
                lines += "f %s\n" % " ".join(face)

            elif i + 1 == F:
                # No newline at the end of the file.
                lines += "f %s" % " ".join(face)

    f.write(lines)

import numpy as np
import torch
from pytorch3d.structures import Meshes


from .pytorch import signed_pow, safe_pow
from .mesh import get_icosphere


def parametric_sq(eta, omega, eps1, eps2):
    cos_eta, sin_eta = signed_pow(torch.cos(eta), eps1), signed_pow(torch.sin(eta), eps1)
    cos_omega, sin_omega = signed_pow(torch.cos(omega), eps2), signed_pow(torch.sin(omega), eps2)
    points = torch.stack([cos_eta * sin_omega, sin_eta, cos_eta * cos_omega], dim=-1)
    return points


def implicit_sq(points, eps1=1, eps2=1, safe=True, as_sdf=False):
    # XXX we only handle the special of eps in [0.1, 2]
    assert torch.all(eps1 >= 0.1) and torch.all(eps1 <= 2)
    assert torch.all(eps2 >= 0.1) and torch.all(eps2 <= 2)
    pow_func = safe_pow if safe else torch.pow
    if safe:
        # we clamp points to [-5, 5] to avoid infinity values obtained by x.pow(20)
        points = points.clamp(-5, 5)

    # XXX iteratively do pow(2) then pow(1/eps) bc pow(float) is not defined on negative values, thus NaN in backward
    x2, y2, z2 = [points[..., k].pow(2) for k in range(3)]
    x, y, z = [pow_func(x2, 1 / eps2), pow_func(y2, 1 / eps1), pow_func(z2, 1 / eps2)]  # not safe bc exp in [0.5, 10]
    res = pow_func(x + z, eps2 / eps1) + y  # not safe because exponent in [0.05, 20]
    if as_sdf:
        # we compute the radial Euclidean distance
        if isinstance(as_sdf, bool):
            return points.norm(dim=-1) * (1 - 1 / (pow_func(res, eps1 / 2) + 1e-6))  # not safe, exp [0.05, 1]
        else:
            # somehow proportional to the radial Euclidean distance
            return pow_func(res, eps1 / 2) - 1  # not safe because exponent in [0.05, 1]
    else:
        return res - 1


def create_sq_meshes(eps1, eps2, scale, level=1):
    N, device = len(eps1), eps1.device
    verts, faces = get_icosphere(level=1).to(device).get_mesh_verts_faces(0)
    eta, omega = torch.asin(verts[..., 1]), torch.atan2(verts[..., 0], verts[..., 2])
    eta, omega = eta[None].expand(N, -1), omega[None].expand(N, -1)
    verts = parametric_sq(eta, omega, eps1, eps2) * scale[:, None]
    return Meshes(verts, faces[None].expand(N, -1, -1))


def sample_sq(eps1, eps2, scale, n_points):
    N, device = len(eps1), eps1.device
    eta = torch.rand(N, n_points, device=device) * np.pi - np.pi/2
    omega = torch.rand(N, n_points, device=device) * 2 * np.pi - np.pi
    cos_eta, sin_eta = signed_pow(torch.cos(eta), eps1), signed_pow(torch.sin(eta), eps1)
    cos_omega, sin_omega = signed_pow(torch.cos(omega), eps2), signed_pow(torch.sin(omega), eps2)
    points = torch.stack([cos_eta * sin_omega, cos_eta * cos_omega, sin_eta], dim=-1)
    return points * scale[:, None]


###########################################
# The following is a vectorized version of showSuperquadrics function from
# https://github.com/bmlklwx/EMS-superquadric_fitting/blob/main/Python/src/EMS/utilities.py
###########################################


def sample_uniform_sq(eps1, eps2, scale, n_points=1000, threshold=1e-2, num_limit=10000, arclength=0.02):
    # avoid numerical instability in sampling
    eps1, eps2 = eps1.clamp(0.01), eps2.clamp(0.01)

    points = []
    for e1, e2, S in zip(eps1, eps2, scale):
        # sampling points in superellipse
        point_eta = uniform_superellipse_sampling(e1, [1, S[2]], threshold, num_limit, arclength)
        point_omega = uniform_superellipse_sampling(e2, [S[0], S[1]], threshold, num_limit, arclength)

        # ellipse product
        point_eta, point_omega = point_eta[:, None, :], point_omega[:, :, None]
        xy = (point_omega * point_eta[0:1])
        z = point_eta[1:2].expand(-1, point_omega.shape[1], -1)
        pc = torch.cat([xy, z], dim=0).view(3, -1).T
        pc = pc[torch.randperm(len(pc))]
        if n_points is not None:
            pc = pc[:n_points]
        points.append(pc)
    return torch.stack(points)


def uniform_superellipse_sampling(epsilon, scale, threshold=1e-2, num_limit=10000, arclength=0.02):
    if isinstance(epsilon, torch.Tensor):
        epsilon = epsilon.item()
    if isinstance(scale[0], torch.Tensor):
        scale[0] = scale[0].item()
    if isinstance(scale[1], torch.Tensor):
        scale[1] = scale[1].item()

    # initialize array storing sampled theta
    theta = np.zeros(num_limit)
    for i in range(num_limit):
        dt = dtheta(theta[i], arclength, threshold, scale, epsilon)
        theta_temp = theta[i] + dt
        if theta_temp > np.pi / 4:
            theta[i + 1] = np.pi / 4
            break
        else:
            if i + 1 < num_limit:
                theta[i + 1] = theta_temp
            else:
                raise Exception(f'Nb sampled points exceed the preset limit {num_limit}, decrease arclength')
    critical = i + 1

    for j in range(critical + 1, num_limit):
        dt = dtheta(theta[j], arclength, threshold, np.flip(scale), epsilon)
        theta_temp = theta[j] + dt
        if theta_temp > np.pi / 4:
            break
        else:
            if j + 1 < num_limit:
                theta[j + 1] = theta_temp
            else:
                raise Exception(f'Nb sampled points exceed the preset limit {num_limit}, decrease arclength')
    num_pt = j
    theta = theta[0 : num_pt + 1]

    point_fw = angle2points(theta[0 : critical + 1], scale, epsilon)
    point_bw = np.flip(angle2points(theta[critical + 1: num_pt + 1], np.flip(scale), epsilon), (0, 1))
    point = np.concatenate((point_fw, point_bw), 1)
    point = np.concatenate((point, np.flip(point[:, 0 : num_pt], 1) * np.array([[-1], [1]]),
                           point[:, 1 : num_pt + 1] * np.array([[-1], [-1]]),
                           np.flip(point[:, 0 : num_pt], 1) * np.array([[1], [-1]])), 1)
    return torch.from_numpy(point)


def dtheta(theta, arclength, threshold, scale, epsilon):
    # calculation the sampling step size
    if theta < threshold:
        dt = np.abs(np.power(arclength / scale[1] + np.power(theta, epsilon), (1 / epsilon)) - theta)
    else:
        dt = arclength / epsilon * ((np.cos(theta) ** 2 * np.sin(theta) ** 2) /
                                    (scale[0] ** 2 * np.cos(theta) ** (2 * epsilon) * np.sin(theta) ** 4 +
                                    scale[1] ** 2 * np.sin(theta) ** (2 * epsilon) * np.cos(theta) ** 4)) ** (1 / 2)
    return dt


def angle2points(theta, scale, epsilon):
    point = np.zeros((2, np.shape(theta)[0]))
    point[0] = scale[0] * np.sign(np.cos(theta)) * np.abs(np.cos(theta)) ** epsilon
    point[1] = scale[1] * np.sign(np.sin(theta)) * np.abs(np.sin(theta)) ** epsilon
    return point

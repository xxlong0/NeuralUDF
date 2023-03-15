#  NeuralWarp  All rights reseved to Thales LAS and ENPC.
#
#  This code is freely available for academic use only and Provided “as is” without any warranty.
#
#  Modification are allowed for academic research provided that the following conditions are met :
#    * Redistributions of source code or any format must retain the above copyright notice and this list of conditions.
#    * Neither the name of Thales LAS and ENPC nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

import numpy as np
import torch


def add_hom(pts):
    try:
        dev = pts.device
        ones = torch.ones(pts.shape[:-1], device=dev).unsqueeze(-1)
        return torch.cat((pts, ones), dim=-1)

    except AttributeError:
        ones = np.ones((pts.shape[0], 1))
        return np.concatenate((pts, ones), axis=1)


def quat_to_rot(q):
    a, b, c, d = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    a2, b2, c2, d2 = a ** 2, b ** 2, c ** 2, d ** 2
    if isinstance(q, torch.Tensor):
        R = torch.empty((q.shape[0], 3, 3))
    else:
        R = np.empty((q.shape[0], 3, 3))
    R[:, 0, 0] = a2 + b2 - c2 - d2
    R[:, 0, 1] = 2 * b * c - 2 * a * d
    R[:, 0, 2] = 2 * a * c + 2 * b * d
    R[:, 1, 0] = 2 * a * d + 2 * b * c
    R[:, 1, 1] = a2 - b2 + c2 - d2
    R[:, 1, 2] = 2 * c * d - 2 * a * b
    R[:, 2, 0] = 2 * b * d - 2 * a * c
    R[:, 2, 1] = 2 * a * b + 2 * c * d
    R[:, 2, 2] = a2 - b2 - c2 + d2

    return R


def rot_to_quat(M):
    q = np.empty((M.shape[0], 4,))
    t = np.trace(M, axis1=1, axis2=2)

    cond1 = t > 0
    cond2 = ~cond1 & (M[:, 0, 0] > M[:, 1, 1]) & (M[:, 0, 0] > M[:, 2, 2])
    cond3 = ~cond1 & ~cond2 & (M[:, 1, 1] > M[:, 2, 2])
    cond4 = ~cond1 & ~cond2 & ~cond3

    S = 2 * np.sqrt(1.0 + t[cond1])
    q[cond1, 0] = 0.25 * S
    q[cond1, 1] = (M[cond1, 2, 1] - M[cond1, 1, 2]) / S
    q[cond1, 2] = (M[cond1, 0, 2] - M[cond1, 2, 0]) / S
    q[cond1, 3] = (M[cond1, 1, 0] - M[cond1, 0, 1]) / S

    S = np.sqrt(1.0 + M[cond2, 0, 0] - M[cond2, 1, 1] - M[cond2, 2,2]) * 2
    q[cond2, 0] = (M[cond2, 2, 1] - M[cond2, 1, 2]) / S
    q[cond2, 1] = 0.25 * S
    q[cond2, 2] = (M[cond2, 0, 1] + M[cond2, 1, 0]) / S
    q[cond2, 3] = (M[cond2, 0, 2] + M[cond2, 2, 0]) / S

    S = np.sqrt(1.0 + M[cond3, 1, 1] - M[cond3, 0, 0] - M[cond3, 2, 2]) * 2
    q[cond3, 0] = (M[cond3, 0, 2] - M[cond3, 2, 0]) / S
    q[cond3, 1] = (M[cond3, 0, 1] + M[cond3, 1, 0]) / S
    q[cond3, 2] = 0.25 * S
    q[cond3, 3] = (M[cond3, 1, 2] + M[cond3, 2, 1]) / S

    S = np.sqrt(1.0 + M[cond4, 2, 2] - M[cond4, 0, 0] - M[cond4, 1, 1]) * 2
    q[cond4, 0] = (M[cond4, 1, 0] - M[cond4, 0, 1]) / S
    q[cond4, 1] = (M[cond4, 0, 2] + M[cond4, 2, 0]) / S
    q[cond4, 2] = (M[cond4, 1, 2] + M[cond4, 2, 1]) / S
    q[cond4, 3] = 0.25 * S

    return q / np.linalg.norm(q, axis=1, keepdims=True)


def normalize(flow, h, w, clamp=None):
    # either h and w are simple float or N torch.tensor where N batch size
    try:
        h.device

    except AttributeError:
        h = torch.tensor(h, device=flow.device).float().unsqueeze(0)
        w = torch.tensor(w, device=flow.device).float().unsqueeze(0)

    if len(flow.shape) == 4:
        w = w.unsqueeze(1).unsqueeze(2)
        h = h.unsqueeze(1).unsqueeze(2)
    elif len(flow.shape) == 3:
        w = w.unsqueeze(1)
        h = h.unsqueeze(1)
    elif len(flow.shape) == 5:
        w = w.unsqueeze(0).unsqueeze(2).unsqueeze(2)
        h = h.unsqueeze(0).unsqueeze(2).unsqueeze(2)

    res = torch.empty_like(flow)
    if res.shape[-1] == 3:
        res[..., 2] = 1

    # for grid_sample with align_corners=True
    # https://github.com/pytorch/pytorch/blob/c371542efc31b1abfe6f388042aa3ab0cef935f2/aten/src/ATen/native/GridSampler.h#L33
    res[..., 0] = 2 * flow[..., 0] / (w - 1) - 1
    res[..., 1] = 2 * flow[..., 1] / (h - 1) - 1

    if clamp:
        return torch.clamp(res, -clamp, clamp)
    else:
        return res


def unnormalize(flow, h, w):
    try:
        h.device
    except AttributeError:
        h = torch.tensor(h, device=flow.device).float().unsqueeze(0)
        w = torch.tensor(w, device=flow.device).float().unsqueeze(0)

    if len(flow.shape) == 4:
        w = w.unsqueeze(1).unsqueeze(2)
        h = h.unsqueeze(1).unsqueeze(2)
    elif len(flow.shape) == 3:
        w = w.unsqueeze(1)
        h = h.unsqueeze(1)

    res = torch.empty_like(flow)

    if res.shape[-1] == 3:
        res[..., 2] = 1

    # idem: https://github.com/pytorch/pytorch/blob/c371542efc31b1abfe6f388042aa3ab0cef935f2/aten/src/ATen/native/GridSampler.h#L33
    res[..., 0] = ((flow[..., 0] + 1) / 2) * (w - 1)
    res[..., 1] = ((flow[..., 1] + 1) / 2) * (h - 1)

    return res
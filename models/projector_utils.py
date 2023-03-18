import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging


def cam2pixel(cam_coords, proj_c2p_rot, proj_c2p_tr, padding_mode, sizeH=None, sizeW=None, with_depth=False):
    """Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 3, H, W]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 3]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
    Returns:
        array of [-1,1] coordinates -- [B, H, W, 2]
    """
    b, _, h, w = cam_coords.size()
    if sizeH is None:
        sizeH = h
        sizeW = w

    cam_coords_flat = cam_coords.view(b, 3, -1)  # [B, 3, H*W]
    if proj_c2p_rot is not None:
        pcoords = proj_c2p_rot.bmm(cam_coords_flat)
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)

    X_norm = 2 * (X / Z) / (sizeW - 1) - 1  # Normalized, -1 if on extreme left,
    # 1 if on extreme right (x = w-1) [B, H*W]
    Y_norm = 2 * (Y / Z) / (sizeH - 1) - 1  # Idem [B, H*W]
    if padding_mode == 'zeros':
        X_mask = ((X_norm > 1) + (X_norm < -1)).detach()
        X_norm[X_mask] = 2  # make sure that no point in warped image is a combinaison of im and gray
        Y_mask = ((Y_norm > 1) + (Y_norm < -1)).detach()
        Y_norm[Y_mask] = 2

    if with_depth:
        pixel_coords = torch.stack([X_norm, Y_norm, Z], dim=2)  # [B, H*W, 3]
        return pixel_coords.view(b, h, w, 3)
    else:
        pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
        return pixel_coords.view(b, h, w, 2)


# ! the correctness of this function have been checked
def sample_ptsFeatures_from_featureMaps(pts, featureMaps, w2cs, intrinsics, WH, proj_matrix=None, return_mask=False,
                                        border=1.0):
    """
    sample features of pts from 2d feature maps
    :param pts: [N_rays, N_samples, 3]
    :param featureMaps: [N_views, C, H, W]
    :param w2cs: [N_views, 4, 4]
    :param intrinsics: [N_views, 3, 3]
    :param proj_matrix: [N_views, 4, 4]
    :param HW:
    :return:
    """
    # normalized to (-1, 1)
    N_rays, n_samples, _ = pts.shape
    N_views = featureMaps.shape[0]

    if proj_matrix is None:
        proj_matrix = torch.matmul(intrinsics[:, :3, :3], w2cs[:, :3, :])

    pts = pts.permute(2, 0, 1).contiguous().view(1, 3, N_rays, n_samples).repeat(N_views, 1, 1, 1)
    pixel_grids = cam2pixel(pts, proj_matrix[:, :3, :3], proj_matrix[:, :3, 3:],
                            'zeros', sizeH=WH[1], sizeW=WH[0])  # (nviews, N_rays, n_samples, 2)

    valid_mask = (torch.abs(pixel_grids[:, :, :, 0]) < border) & (
            torch.abs(pixel_grids[:, :, :, 1]) < border)  # (nviews, N_rays, n_samples)

    pts_feature = F.grid_sample(featureMaps, pixel_grids,
                                padding_mode='zeros',
                                align_corners=True)  # [N_views, C, N_rays, n_samples]

    if return_mask:
        return pts_feature, valid_mask
    else:
        return pts_feature

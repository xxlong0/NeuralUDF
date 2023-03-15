import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
# import mcubes
# import trimesh
# from icecream import ic

import pdb


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


def convert_to_ndc_coordinate(point_samples, dirs, w2c, intrinsic, inv_scale, near=2, far=6, pad_scale=0.5):
    """
    convert points (in world space or camera space) to normalized device coordinate of ref image
    :param w2cs: [4,4]
    :param intrinsics:  [, 3]
    :param point_samples: [N_rays, n_samples, 3]
    :param inv_scale: [W-1, H-1] used to normalize pixel coordinates
    :param near: batch of float number or 3-array (point of bounding box) [b, 1] or [b, 3]
    :param far:
    :param pad_scale: the pts of source images maybe out of ref image after reprojection
    :return:
    """

    N_rays, n_samples = point_samples.shape[:2]

    # convert to camera space
    if w2c is not None:
        point_samples = (torch.matmul(w2c[None, None, :3, :3], point_samples[:, :, :, None]) \
                         + w2c[None, None, :3, 3:]).squeeze(-1)  # (N_rays, n_samples, 3)

    if intrinsic is not None:
        # using projection
        point_samples_pixel = (torch.matmul(intrinsic[None, None, :3, :3],
                                            point_samples[:, :, :, None])).squeeze(
            -1)  # (N_rays, n_samples, 3)

        # todo: the uv coordinates might be out of range (0, 1), for pixels of source images
        point_samples_pixel[:, :, :2] = point_samples_pixel[:, :, :2] / (
                point_samples_pixel[:, :, 2:] * inv_scale[None, None, :])  # normalize to 0~1
        point_samples_pixel[:, :, :2] = point_samples_pixel[:, :, :2] * pad_scale  # rescale

        point_samples_pixel[:, :, 2] = (point_samples_pixel[:, :, 2] - near[None, None, :]) \
                                       / (far[None, None, :] - near[None, None, :])  # normalize to 0~1
    else:
        # using bounding box
        point_samples_pixel = (point_samples - near[:, None, None, 3]) / (
                far[:, None, None, 3] - near[:, None, None, 3])  # normalize to 0~1
    del point_samples

    dirs_ndc = dirs @ w2c[:3, :3].t()  # [N_rays, 3]

    return point_samples_pixel, dirs_ndc  # (N_rays, n_samples, 3)


# - checked the correctness
def sample_ptsFeatures_from_featureVolume(pts, featureVolume, vol_dims=None, partial_vol_origin=None, vol_size=None):
    """
    sample feature of pts_wrd from featureVolume, all in world space
    :param pts: [N_rays, n_samples, 3]
    :param featureVolume: [C,wX,wY,wZ]
    :param vol_dims: [3] "3" for dimX, dimY, dimZ
    :param partial_vol_origin: [3]
    :return: pts_feature: [N_rays, n_samples, C]
    :return: valid_mask: [N_rays]
    """

    N_rays, n_samples, _ = pts.shape

    if vol_dims is None:
        pts_normalized = pts
    else:
        # normalized to (-1, 1)
        pts_normalized = 2 * (pts - partial_vol_origin[None, None, :]) / (vol_size * (vol_dims[None, None, :] - 1)) - 1

    valid_mask = (torch.abs(pts_normalized[:, :, 0]) < 1.0) & (
            torch.abs(pts_normalized[:, :, 1]) < 1.0) & (
                         torch.abs(pts_normalized[:, :, 2]) < 1.0)  # (N_rays, n_samples)

    pts_normalized = torch.flip(pts_normalized, dims=[-1])  # ! reverse the xyz for grid_sample

    # ! checked grid_sample, (x,y,z) is for (D,H,W), reverse for (W,H,D)
    pts_feature = F.grid_sample(featureVolume[None, :, :, :, :], pts_normalized[None, None, :, :, :],
                                padding_mode='zeros',
                                align_corners=True).view(-1, N_rays, n_samples)  # [C, N_rays, n_samples]

    # onesVolume = torch.ones([1, *featureVolume.shape[1:]]).to(featureVolume.dtype).to(featureVolume.device)
    # pts_valid = F.grid_sample(onesVolume[None, :, :, :, :], pts_normalized[None, None, :, :, :], padding_mode='zeros',
    #                           align_corners=True).view(-1, N_rays, n_samples)  # [1, N_rays, n_samples]
    #
    # # the sampled pts of one ray may be out of range of featureVolume
    # valid_mask = torch.mean(pts_valid.squeeze(0), dim=1, keepdim=False) > 0.6

    pts_feature = pts_feature.permute(1, 2, 0)  # [N_rays, n_samples, C]
    return pts_feature, valid_mask


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

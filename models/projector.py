import torch
import torch.nn.functional as F
from models.projector_utils import cam2pixel


class Projector():
    def project(self, pts, c2ws, intrinsics, img_wh, color_maps=None):
        """

        Parameters
        ----------
        pts : [N_rays, n_samples, 3]
        c2ws : [N_views, 4, 4]
        intrinsics : [N_views, 3, 3]
        img_wh : [2]
        color_maps : [N_views, 3, H, W]

        Returns
        -------
        proj_xys: [N_views, N_rays, nsamples, 2]
        proj_zs:
        proj_mask:
        proj_colors:
        """
        device = pts.device
        sizeW = img_wh[0]
        sizeH = img_wh[1]

        # normalized to (-1, 1)
        N_rays, n_samples, _ = pts.shape
        w2cs = torch.inverse(c2ws)
        N_views = w2cs.shape[0]

        proj_matrix = torch.matmul(intrinsics[:, :3, :3], w2cs[:, :3, :])

        pts = pts.permute(2, 0, 1).contiguous().view(1, 3, N_rays, n_samples).repeat(N_views, 1, 1, 1)
        pixel_grids = cam2pixel(pts, proj_matrix[:, :3, :3], proj_matrix[:, :3, 3:],
                                'zeros', sizeH=img_wh[1], sizeW=img_wh[0],
                                with_depth=True)  # (nviews, N_rays, n_samples, 3)
        proj_uvs = pixel_grids[:, :, :, :2]
        proj_zs = pixel_grids[:, :, :, 2]

        proj_mask = (torch.abs(proj_uvs[:, :, :, 0]) < 1.0) & (
                torch.abs(proj_uvs[:, :, :, 1]) < 1.0)  # (nviews, N_rays, n_samples)

        proj_colors = None
        if color_maps is not None:
            proj_colors = F.grid_sample(color_maps, proj_uvs,
                                        padding_mode='zeros',
                                        align_corners=True)  # [N_views, C, N_rays, n_samples]
        X_norm = proj_uvs[:, :, :, :1]
        Y_norm = proj_uvs[:, :, :, 1:2]
        X = (X_norm + 1) * 0.5 * (sizeW - 1)
        Y = (Y_norm + 1) * 0.5 * (sizeH - 1)

        proj_xys = torch.cat([X, Y], dim=-1)

        return proj_xys, proj_zs, proj_mask, proj_colors

    def generate_rays(self, pixels_xys, intrinsics, c2ws):
        """

        Parameters
        ----------
        pixels_xys : [N_views, N_num, 2]
        intrinsics : [N_views, 3, 3]
        c2ws : [N_views, 4, 4]

        Returns
        -------

        """
        intrinsics_inv = torch.inverse(intrinsics)
        p = torch.cat([pixels_xys, torch.ones_like(pixels_xys[:, :, :1])], dim=-1)  # N_views, n, 3
        p = torch.matmul(intrinsics_inv[:, None, :3, :3], p[:, :, :, None]).squeeze()  # N_views, n, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # N_views, n, 3
        rays_v = torch.matmul(c2ws[:, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # N_views, n, 3
        rays_o = c2ws[:, None, :3, 3].expand(rays_v.shape)  # N_views, n, 3
        return rays_o, rays_v

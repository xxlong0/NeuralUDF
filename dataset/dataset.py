import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
from glob import glob
# from icecream import ic
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
import pdb


# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


# deepfashion k (293067.6,293067.6,511.5,511.5)

class Dataset:
    def __init__(self, conf):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf
        self.dataset_name = self.conf.get_string('dataset_name', default='dtu')

        # self.data_type = self.conf.get_string('data_type', default='dtu')  # dtu or bmvs

        self.data_dir = conf.get_string('data_dir')
        self.render_cameras_name = conf.get_string('render_cameras_name')
        self.object_cameras_name = conf.get_string('object_cameras_name')

        self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
        self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)

        self.downsample_factor = conf.get_float('downsample_factor', default=1.0)

        camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
        self.camera_dict = camera_dict
        if self.dataset_name == 'dtu' or self.dataset_name == 'deepfashion3d':
            self.images_lis = sorted(glob(os.path.join(self.data_dir, 'image/*.png')))
            self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*.png')))
        elif self.dataset_name == 'bmvs':
            self.images_lis = sorted(glob(os.path.join(self.data_dir, 'blended_images/*.jpg')))
            self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'masks/*.jpg')))
        self.n_images = len(self.images_lis)

        self.images_np = np.stack([cv.imread(im_name) for im_name in self.images_lis]) / 256.0
        self.masks_np = np.stack([cv.imread(im_name) for im_name in self.masks_lis]) / 256.0

        # world_mat is a projection matrix from world to image
        self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.intrinsics_all = []
        self.pose_all = []

        for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)

            # rescale intrinsics
            intrinsics[:2] *= self.downsample_factor

            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        self.images = torch.from_numpy(self.images_np.astype(np.float32)).cuda()  # [n_images, H, W, 3]
        self.masks = torch.from_numpy(self.masks_np.astype(np.float32)).cuda()  # [n_images, H, W, 3]
        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)  # [n_images, 4, 4]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = torch.stack(self.pose_all).to(self.device)  # [n_images, 4, 4]

        # rescale images and masks
        if self.downsample_factor != 1:
            self.images = F.interpolate(self.images.permute(0, 3, 1, 2).contiguous(), size=None,
                                        scale_factor=self.downsample_factor,
                                        mode='bilinear').permute(0, 2, 3, 1).contiguous().cuda()

            self.masks = F.interpolate(self.masks.permute(0, 3, 1, 2).contiguous(), size=None,
                                       scale_factor=self.downsample_factor,
                                       mode='bilinear').permute(0, 2, 3, 1).contiguous()

        self.H, self.W = self.images.shape[1], self.images.shape[2]
        self.image_pixels = self.H * self.W

        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([1.01, 1.01, 1.01, 1.0])
        # if self.dataset_name != 'dtu':
        #     object_bbox_min = object_bbox_min * 0.6
        #     object_bbox_max = object_bbox_max * 0.6

        # Object scale mat: region of interest to **extract mesh**
        object_scale_mat = np.load(os.path.join(self.data_dir, self.object_cameras_name))['scale_mat_0']
        object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
        object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
        self.object_bbox_min = object_bbox_min[:3, 0]
        self.object_bbox_max = object_bbox_max[:3, 0]

        self.ref_src_pair = self.prepare_ref_src_pairs()

        print('Load data: End')

    def prepare_ref_src_pairs(self):
        # prepare the pairs of a reference image and several supporting images for color blending
        ref_src_pairs = {}
        cam_loc = self.pose_all[:, :3, 3]  # [V, 3]
        pair_dist = torch.cdist(cam_loc[None, :, :], cam_loc[None, :, :], p=2.0)  # [1, V, V]
        sorted_pdist, indices = torch.sort(pair_dist, descending=False, dim=2)
        for i in range(self.n_images):
            ref_src_pairs[i] = indices[0, i][1:10]

        print(ref_src_pairs)
        return ref_src_pairs

    def get_ref_src_info(self, img_idx, num=8):
        if isinstance(img_idx, torch.Tensor):
            img_idx = img_idx.cpu().numpy().item()
        src_idx = self.ref_src_pair[img_idx][:num]
        ref_c2w = self.pose_all[img_idx]
        c2ws = self.pose_all[src_idx]
        images = self.images[src_idx]
        intrinsics = self.intrinsics_all[src_idx]
        return ref_c2w.cuda(), c2ws.cuda(), intrinsics.cuda(), images.cuda().permute(0, 3, 1, 2), [self.W, self.H]

    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def gen_one_ray_at(self, img_idx, x, y):
        """

        Parameters
        ----------
        img_idx :
        x : for width
        y : for height

        Returns
        -------

        """
        image = np.uint8(self.images_np[img_idx] * 256)
        image = cv.resize(image, fx=self.downsample_factor, fy=self.downsample_factor, dsize=None)
        image2 = cv.circle(image, (x, y), radius=10, color=(0, 0, 255), thickness=-1)

        pixels_x = torch.Tensor([x]).long()
        pixels_y = torch.Tensor([y]).long()
        color = self.images[img_idx][(pixels_y, pixels_x)]  # batch_size, 3
        mask = (self.masks[img_idx][(pixels_y, pixels_x)] > 0).to(torch.float32)  # batch_size, 3
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze(-1)  # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze(-1)  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape)  # batch_size, 3
        return torch.cat([rays_o.cuda(), rays_v.cuda(), color.cuda(), mask[:, :1].cuda()],
                         dim=-1), image2  # batch_size, 10

    def gen_random_rays_at(self, img_idx, batch_size, importance_sample=False):
        """
        Generate random rays at world space from one camera.
        """

        if not importance_sample:
            pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
            pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        elif importance_sample and self.masks is not None:  # sample more pts in the valid mask regions
            pixels_x_1 = torch.randint(low=0, high=self.W, size=[batch_size // 4])
            pixels_y_1 = torch.randint(low=0, high=self.H, size=[batch_size // 4])

            ys, xs = torch.meshgrid(torch.linspace(0, self.H - 1, self.H),
                                    torch.linspace(0, self.W - 1, self.W))  # pytorch's meshgrid has indexing='ij'
            p = torch.stack([xs, ys], dim=-1)  # H, W, 2
            p_valid = p[self.masks[img_idx][:, :, 0] > 0]  # [num, 2]
            random_idx = torch.randint(low=0, high=p_valid.shape[0], size=[batch_size // 4 * 3])
            p_select = p_valid[random_idx]  # [N_rays//2, 2]
            pixels_x_2 = p_select[:, 0]
            pixels_y_2 = p_select[:, 1]

            pixels_x = torch.cat([pixels_x_1, pixels_x_2], dim=0).to(torch.int64)
            pixels_y = torch.cat([pixels_y_1, pixels_y_2], dim=0).to(torch.int64)

        color = self.images[img_idx][(pixels_y, pixels_x)]  # batch_size, 3
        mask = (self.masks[img_idx][(pixels_y, pixels_x)] > 0).to(torch.float32)  # batch_size, 3
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze()  # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape)  # batch_size, 3
        return torch.cat([rays_o.cuda(), rays_v.cuda(), color.cuda(), mask[:, :1].cuda()], dim=-1)  # batch_size, 10

    def gen_random_rays_patches_at(self, img_idx, batch_size, importance_sample=False, h_patch_size=3,
                                   crop_patch=False):
        """
        Generate random rays at world space from one camera.
        """

        if not importance_sample:
            pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
            pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        elif importance_sample and self.masks is not None:  # sample more pts in the valid mask regions
            pixels_x_1 = torch.randint(low=0, high=self.W, size=[batch_size // 4])
            pixels_y_1 = torch.randint(low=0, high=self.H, size=[batch_size // 4])

            ys, xs = torch.meshgrid(torch.linspace(0, self.H - 1, self.H),
                                    torch.linspace(0, self.W - 1, self.W))  # pytorch's meshgrid has indexing='ij'
            p = torch.stack([xs, ys], dim=-1)  # H, W, 2
            p_valid = p[self.masks[img_idx][:, :, 0] > 0]  # [num, 2]
            random_idx = torch.randint(low=0, high=p_valid.shape[0], size=[batch_size // 4 * 3])
            p_select = p_valid[random_idx]  # [N_rays//2, 2]
            pixels_x_2 = p_select[:, 0]
            pixels_y_2 = p_select[:, 1]

            pixels_x = torch.cat([pixels_x_1, pixels_x_2], dim=0).to(torch.int64)
            pixels_y = torch.cat([pixels_y_1, pixels_y_2], dim=0).to(torch.int64)

        patch_color, patch_mask = None, None
        if crop_patch:
            # - crop patch from images
            offsets = build_patch_offset(h_patch_size)
            grid_patch = torch.stack([pixels_x, pixels_y], dim=-1).view(-1, 1, 2) + offsets.float()  # [N_pts, Npx, 2]
            patch_mask = (pixels_x > h_patch_size) * (pixels_x < (self.W - h_patch_size)) * (
                    pixels_y > h_patch_size) * (
                                 pixels_y < self.H - h_patch_size)  # [N_pts]
            grid_patch_u = 2 * grid_patch[:, :, 0] / (self.W - 1) - 1
            grid_patch_v = 2 * grid_patch[:, :, 1] / (self.H - 1) - 1
            grid_patch_uv = torch.stack([grid_patch_u, grid_patch_v], dim=-1)  # [N_pts, Npx, 2]
            patch_color = \
                F.grid_sample(self.images[img_idx][None, :, :, :].cuda().permute(0, 3, 1, 2),
                              grid_patch_uv[None, :, :, :], mode='bilinear',
                              padding_mode='zeros')[0]  # [3, N_pts, Npx]
            patch_color = patch_color.permute(1, 2, 0).contiguous().cuda()
            patch_mask = patch_mask.view(-1, 1).cuda()

        # normalized ndc uv coordinates, (-1, 1)
        ndc_u = 2 * pixels_x / (self.W - 1) - 1
        ndc_v = 2 * pixels_y / (self.H - 1) - 1
        rays_ndc_uv = torch.stack([ndc_u, ndc_v], dim=-1).view(-1, 2).float()

        color = self.images[img_idx][(pixels_y, pixels_x)]  # batch_size, 3
        mask = (self.masks[img_idx][(pixels_y, pixels_x)] > 0).to(torch.float32)  # batch_size, 3
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze()  # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape)  # batch_size, 3

        rays = torch.cat([rays_o.cuda(), rays_v.cuda(), color.cuda(), mask[:, :1].cuda()], dim=-1)  # batch_size, 10

        sample = {
            'rays': rays,
            'rays_ndc_uv': rays_ndc_uv.cuda(),
            'rays_norm_XYZ_cam': p.cuda(),  # - XYZ_cam, before multiply depth,
            'rays_patch_color': patch_color,
            'rays_patch_mask': patch_mask
        }

        return sample

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
        pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
        pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = torch.from_numpy(pose[:3, :3]).cuda()
        trans = torch.from_numpy(pose[:3, 3]).cuda()
        rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d ** 2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def image_at(self, idx, resolution_level):
        img = cv.imread(self.images_lis[idx])
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)


def build_patch_offset(h_patch_size):
    offsets = torch.arange(-h_patch_size, h_patch_size + 1)
    return torch.stack(torch.meshgrid(offsets, offsets)[::-1], dim=-1).view(1, -1, 2)  # nb_pixels_patch * 2

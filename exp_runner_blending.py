import torch
import torch.nn.functional as F
import argparse
import os
import logging
import numpy as np
import cv2 as cv
import trimesh
from shutil import copyfile
from torch.utils.tensorboard import SummaryWriter
from icecream import ic
from tqdm import tqdm
from pyhocon import ConfigFactory, HOCONConverter
from dataset.dataset import Dataset
from models.fields import ResidualRenderingNetwork
from models.fields import SDFNetwork, UDFNetwork, BetaNetwork
from models.fields import SingleVarianceNetwork
from models.fields import NeRF
from models.udf_renderer_blending import UDFRendererBlending, extract_fields, extract_gradient_fields

from loss.loss import ColorLoss

from termcolor import colored

import h5py

from extract_mesh import get_mesh_udf_fast

import matplotlib.pyplot as plt


class Runner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', model_type='', is_continue=False, args=None):

        # Initial setting
        self.device = torch.device('cuda')

        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)

        # modify the setting based on input
        if args.learning_rate > 0:
            self.conf['train']['learning_rate'] = args.learning_rate
        if args.learning_rate_geo > 0:
            self.conf['train']['learning_rate_geo'] = args.learning_rate_geo
        if args.sparse_weight > 0:
            self.conf['train']['sparse_weight'] = args.sparse_weight

        self.base_exp_dir = os.path.join(self.conf['general.base_exp_dir'], self.conf['general.expname'])
        os.makedirs(self.base_exp_dir, exist_ok=True)

        self.dataset_name = self.conf.get_string('dataset.dataset_name', default='general')
        self.dataset = Dataset(self.conf['dataset'])

        self.iter_step = 0

        # trainning parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')

        # setting about learning rate schedule
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_geo = self.conf.get_float('train.learning_rate_geo')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)
        # don't train the udf network in the early steps
        self.fix_geo_end = self.conf.get_float('train.fix_geo_end', default=500)
        self.reg_weights_schedule = args.reg_weights_schedule
        self.warmup_sample = self.conf.get_bool('train.warmup_sample', default=False)  # * training schedule
        # whether the udf network and appearance network share the same learning rate
        self.same_lr = self.conf.get_bool('train.same_lr', default=False)

        # weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.igr_ns_weight = self.conf.get_float('train.igr_ns_weight', default=0.0)
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.sparse_weight = self.conf.get_float('train.sparse_weight', default=0.0)

        # loss functions
        self.color_loss_func = ColorLoss(**self.conf['color_loss'])
        self.color_base_weight = self.conf.get_float('color_loss.color_base_weight', 0.0)
        self.color_weight = self.conf.get_float('color_loss.color_weight', 0.0)
        self.color_pixel_weight = self.conf.get_float('color_loss.color_pixel_weight', 0.0)
        self.color_patch_weight = self.conf.get_float('color_loss.color_patch_weight', 0.0)

        self.is_continue = is_continue
        self.is_finetune = args.is_finetune

        self.vis_ray = args.vis_ray  # visualize a ray for debug

        self.mode = mode
        self.model_type = self.conf['general.model_type']
        if model_type != '':  # overwrite
            self.model_type = model_type
        self.model_list = []
        self.writer = None

        # Networks
        params_to_train = []
        params_to_train_nerf = []
        params_to_train_geo = []

        self.nerf_outside = None
        self.nerf_coarse = None
        self.nerf_fine = None
        self.sdf_network_fine = None
        self.udf_network_fine = None
        self.variance_network_fine = None
        self.color_network_coarse = None
        self.color_network_fine = None

        self.nerf_outside = NeRF(**self.conf['model.nerf']).to(self.device)
        self.udf_network_fine = UDFNetwork(**self.conf['model.udf_network']).to(self.device)
        self.variance_network_fine = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.color_network_fine = ResidualRenderingNetwork(**self.conf['model.rendering_network']).to(self.device)
        self.beta_network = BetaNetwork(**self.conf['model.beta_network']).to(self.device)
        params_to_train_nerf += list(self.nerf_outside.parameters())
        params_to_train_geo += list(self.udf_network_fine.parameters())
        params_to_train += list(self.variance_network_fine.parameters())
        params_to_train += list(self.color_network_fine.parameters())
        params_to_train += list(self.beta_network.parameters())

        self.optimizer = torch.optim.Adam(
            [{'params': params_to_train_geo, 'lr': self.learning_rate_geo}, {'params': params_to_train},
             {'params': params_to_train_nerf}],
            lr=self.learning_rate)

        self.renderer = UDFRendererBlending(self.nerf_outside,
                                            self.udf_network_fine,
                                            self.variance_network_fine,
                                            self.color_network_fine,
                                            self.beta_network,
                                            **self.conf['model.udf_renderer'])

        # Load checkpoint
        latest_model_name = None
        if is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth':
                    # if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        if self.mode[:5] == 'train':
            self.file_backup()

    def update_learning_rate(self, start_g_id=0):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups[start_g_id:]:
            g['lr'] = self.learning_rate * learning_factor

    def update_learning_rate_geo(self):
        if self.iter_step < self.fix_geo_end:  # * make bg nerf learn first
            learning_factor = 0.0
        elif self.iter_step < self.warm_up_end * 2:
            learning_factor = self.iter_step / (self.warm_up_end * 2)
        elif self.iter_step < self.end_iter * 0.5:
            learning_factor = 1.0
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.end_iter * 0.5) / (self.end_iter - self.end_iter * 0.5)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups[:1]:
            g['lr'] = self.learning_rate_geo * learning_factor

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def regularization_weights_schedule(self):
        igr_ns_weight = 0.0
        sparse_weight = 0.0

        end1 = self.end_iter // 5
        end2 = self.end_iter // 2

        if self.iter_step >= end1:
            igr_ns_weight = self.igr_ns_weight * np.clip((self.iter_step - end1) / end1, 0.0, 1.0)
        if self.iter_step >= end2:
            sparse_weight = self.sparse_weight

        return igr_ns_weight, sparse_weight

    def train(self):
        self.train_udf()

    def get_flip_saturation(self, flip_saturation_max=0.9):
        start = 10000
        if self.iter_step < start:
            flip_saturation = 0.0
        elif self.iter_step < self.end_iter * 0.5:
            flip_saturation = flip_saturation_max
        else:
            flip_saturation = 1.0

        if self.is_finetune:
            flip_saturation = 1.0

        return flip_saturation

    def adjust_color_loss_weights(self):
        if self.is_finetune:
            factor = 1.0
        else:
            if self.iter_step < 10000:
                factor = 0
            elif self.iter_step < 20000:
                factor = np.clip((self.iter_step - 10000) / 10000, 0, 1)
            else:
                factor = 1.

        if self.color_base_weight < self.color_weight:
            color_base_weight = self.color_base_weight * factor
        else:
            color_base_weight = self.color_base_weight
        color_weight = self.color_weight
        color_pixel_weight = self.color_pixel_weight * factor
        color_patch_weight = self.color_patch_weight * factor

        self.color_loss_func.set_color_weights(color_base_weight, color_weight, color_pixel_weight, color_patch_weight)

        return color_base_weight, color_weight, color_pixel_weight, color_patch_weight

    def train_udf(self):
        image_perm = torch.randperm(self.dataset.n_images)
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        res_step = self.end_iter - self.iter_step

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate

        beta_flag = True
        for iter_i in tqdm(range(res_step)):

            if self.same_lr:
                self.update_learning_rate(start_g_id=0)
            else:
                self.update_learning_rate(start_g_id=1)
                self.update_learning_rate_geo()

            color_base_weight, color_weight, color_pixel_weight, color_patch_weight = self.adjust_color_loss_weights()

            img_idx = image_perm[self.iter_step % len(image_perm)]
            sample = self.dataset.gen_random_rays_patches_at(
                img_idx, self.batch_size,
                crop_patch=color_patch_weight > 0.0, h_patch_size=self.color_loss_func.h_patch_size)

            data = sample['rays']
            rays_uv = sample['rays_ndc_uv']
            gt_patch_colors = sample['rays_patch_color']
            gt_patch_mask = sample['rays_patch_mask']

            if color_pixel_weight > 0. or color_patch_weight > 0.:
                # todo: this load is very slow
                ref_c2w, src_c2ws, src_intrinsics, src_images, img_wh = self.dataset.get_ref_src_info(img_idx)
                src_w2cs = torch.inverse(src_c2ws)
            else:
                ref_c2w, src_c2ws, src_w2cs, src_intrinsics, src_images = None, None, None, None, None

            # todo load supporting images

            rays_o, rays_d, true_rgb, mask = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10]

            near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)

            mask = (mask > 0.5).float()

            mask_sum = mask.sum() + 1e-5

            render_out = self.renderer.render(rays_o, rays_d, near, far,
                                              flip_saturation=self.get_flip_saturation(),
                                              color_maps=src_images if color_pixel_weight > 0. else None,
                                              w2cs=src_w2cs,
                                              intrinsics=src_intrinsics,
                                              query_c2w=ref_c2w,
                                              img_index=None,
                                              rays_uv=rays_uv if color_patch_weight > 0 else None,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio())

            weight_sum = render_out['weight_sum']

            color_base = render_out['color_base']
            color = render_out['color']
            color_pixel = render_out['color_pixel']
            patch_colors = render_out['patch_colors']
            patch_mask = (render_out['patch_mask'].float()[:, None] * (weight_sum > 0.5).float()) > 0. \
                if render_out['patch_mask'] is not None else None
            pixel_mask = mask if self.mask_weight > 0 else None

            variance = render_out['variance']
            beta = render_out['beta']
            gamma = render_out['gamma']

            gradient_error = render_out['gradient_error']
            gradient_error_near_surface = render_out['gradient_error_near_surface']
            sparse_error = render_out['sparse_error']

            udf = render_out['udf']
            udf_min = udf.min(dim=1)[0][mask[:, 0] > 0.5].mean()

            color_losses = self.color_loss_func(
                color_base, color, true_rgb, color_pixel,
                pixel_mask, patch_colors, gt_patch_colors, patch_mask
            )

            color_total_loss = color_losses['loss']
            color_base_loss = color_losses['color_base_loss']
            color_loss = color_losses['color_loss']
            color_pixel_loss = color_losses['color_pixel_loss']
            color_patch_loss = color_losses['color_patch_loss']

            psnr = 20.0 * torch.log10(
                1.0 / (((color - true_rgb) ** 2 * mask).sum() / (mask_sum * 3.0)).sqrt())

            # mask loss
            # mask_loss = (weight_sum - mask).abs().mean()
            mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)

            # Eikonal loss
            gradient_error_loss = gradient_error

            mask_weight = self.mask_weight

            if variance.mean() < 2 * beta.item() and variance.mean() < 0.01 and beta_flag and self.variance_network_fine.variance.requires_grad:
                print("make beta trainable")
                self.beta_network.set_beta_trainable()
                beta_flag = False

            if self.variance_network_fine.variance.requires_grad is False and self.iter_step > 20000:
                self.variance_network_fine.set_trainable()

            if not self.reg_weights_schedule:
                igr_ns_weight = self.igr_ns_weight
                sparse_weight = self.sparse_weight
            else:
                igr_ns_weight, sparse_weight = self.regularization_weights_schedule()

            loss = color_total_loss + \
                   mask_loss * mask_weight + \
                   gradient_error_near_surface * igr_ns_weight + \
                   sparse_error * sparse_weight + \
                   gradient_error_loss * self.igr_weight

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_step += 1

            self.writer.add_scalar('Loss/loss', loss, self.iter_step)
            self.writer.add_scalar('Loss/mask_loss', mask_loss, self.iter_step)
            self.writer.add_scalar('Loss/gradient_error_loss', gradient_error_loss, self.iter_step)
            self.writer.add_scalar('Sta/variance', variance.mean(), self.iter_step)
            self.writer.add_scalar('Sta/beta', beta.item(), self.iter_step)
            self.writer.add_scalar('Sta/psnr', psnr, self.iter_step)

            if self.iter_step % self.report_freq == 0:
                print(self.base_exp_dir)
                print('iter:{:8>d} loss = {:.4f} '
                      'color_total_loss = {:.4f} '
                      'eki_loss = {:.4f} '
                      'eki_ns_loss = {:.4f} '
                      'mask_loss = {:.4f} '
                      'sparse_loss = {:.4f} '.format(self.iter_step, loss, color_total_loss, gradient_error_loss,
                                                     gradient_error_near_surface,
                                                     mask_loss,
                                                     sparse_error))
                print('iter:{:8>d} c_base_loss = {:.4f} '
                      'color_loss = {:.4f} '
                      'c_pixel_loss = {:.4f} '
                      'c_patch_loss = {:.4f} '.format(self.iter_step, color_base_loss, color_loss, color_pixel_loss,
                                                      color_patch_loss))
                print('iter:{:8>d} '
                      'variance = {:.6f} '
                      'beta = {:.6f} '
                      'gamma = {:.4f} '
                      'lr_geo={:.8f} lr={:.8f} '.format(self.iter_step,
                                                        variance.mean(), beta.item(), gamma.item(),
                                                        self.optimizer.param_groups[0]['lr'],
                                                        self.optimizer.param_groups[1]['lr']))

                print(colored('psnr = {:.4f} '
                              'weight_sum = {:.4f} '
                              'weight_sum_fg_bg = {:.4f} '
                              'udf_min = {:.8f} '
                              'udf_mean = {:.4f} '
                              'mask_weight = {:.4f} '
                              'sparse_weight = {:.4f} '
                              'igr_ns_weight = {:.4f} '
                              'igr_weight = {:.4f} '.format(psnr, (render_out['weight_sum'] * mask).sum() / mask_sum,
                                                            (render_out['weight_sum_fg_bg'] * mask).sum() / mask_sum,
                                                            udf_min, udf.mean(), mask_weight, sparse_weight,
                                                            igr_ns_weight,
                                                            self.igr_weight,
                                                            ), 'green'))

                ic(self.get_flip_saturation())

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()

            if self.dataset_name == 'general':
                if self.iter_step % self.val_freq == 0:
                    self.validate()

                if self.iter_step % (self.val_mesh_freq * 2) == 0 and self.vis_ray:
                    for i in range(-self.dataset.H // 4, self.dataset.H // 4, 20):
                        self.visualize_one_ray(img_idx=33, px=self.dataset.W // 2, py=self.dataset.H // 2 + i)

            if self.iter_step % self.val_mesh_freq == 0:
                self.validate_mesh(threshold=args.threshold)
                try:
                    self.extract_udf_mesh(world_space=True, dist_threshold_ratio=2.0)
                except:
                    print("extract udf mesh fails")

            if self.iter_step % len(image_perm) == 0:
                image_perm = torch.randperm(self.dataset.n_images)

    def file_backup(self):
        # copy python file
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        # copy configs
        # copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))
        with open(os.path.join(self.base_exp_dir, 'recording', 'config.conf'), "w") as fd:
            res = HOCONConverter.to_hocon(self.conf)
            fd.write(res)

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name),
                                map_location=self.device)

        self.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.udf_network_fine.load_state_dict(checkpoint['udf_network_fine'])
        self.variance_network_fine.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network_fine.load_state_dict(checkpoint['color_network_fine'])
        self.beta_network.load_state_dict(checkpoint['beta_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']

        if self.is_finetune:
            self.iter_step = 0

        logging.info('End')

    def save_checkpoint(self):
        checkpoint = None

        checkpoint = {
            'nerf': self.nerf_outside.state_dict(),
            'udf_network_fine': self.udf_network_fine.state_dict(),
            'variance_network_fine': self.variance_network_fine.state_dict(),
            'color_network_fine': self.color_network_fine.state_dict(),
            'beta_network': self.beta_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step, }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint,
                   os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def visualize_one_ray(self, img_idx, px, py):
        """
        Visualize the udf values of a ray
        Parameters
        ----------
        idx : the image idx
        px : for width
        py : for height

        Returns
        -------

        """

        background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

        data, image = self.dataset.gen_one_ray_at(img_idx, px, py)

        image = cv.resize(image, fx=0.25, fy=0.25, dsize=None)

        data_c = self.dataset.gen_random_rays_at(img_idx, 512)

        near, far = self.dataset.near_far_from_sphere(data_c[:, :3], data_c[:, 3: 6])

        rays_o, rays_d, true_rgb, mask = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10]

        render_out = self.renderer.render(rays_o, rays_d, near[:1, :], far[:1, :],
                                          cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                          background_rgb=background_rgb,
                                          flip_saturation=self.get_flip_saturation())

        udf = render_out['udf'][0].detach().cpu().numpy()

        z_vals = render_out['mid_z_vals'][0].detach().cpu().numpy() - near[0, 0].cpu().numpy()
        n_samples = z_vals.shape[0]

        depth = render_out['depth'][0].detach().cpu().numpy().item() - near[0, 0].cpu().numpy()

        gradient_mag = render_out['gradient_mag'][0].detach().cpu().numpy()
        true_cos = render_out['true_cos'][0].detach().cpu().numpy()
        weights = render_out['weights'][0].detach().cpu().numpy()[:z_vals.shape[0]]
        if render_out['vis_prob'] is not None:
            vis_prob = render_out['vis_prob'][0].detach().cpu().numpy()
        else:
            vis_prob = None
        alpha = render_out['alpha'][0].detach().cpu().numpy()
        alpha_plus = render_out['alpha_plus'][0].detach().cpu().numpy()
        alpha_minus = render_out['alpha_minus'][0].detach().cpu().numpy()
        alpha_occ = render_out['alpha_occ'][0].detach().cpu().numpy()
        raw_occ = render_out['raw_occ'][0].detach().cpu().numpy()
        # dists = render_out['dists'][0].detach().cpu().numpy()

        depth_min = depth - 2 / 512. * 10
        depth_max = depth + 2 / 512. * 10

        # z_vals = z_vals - near[0,0].cpu().numpy()

        print(depth)

        start_idx = np.argmin(np.abs(z_vals - depth_min))
        end_idx = np.argmin(np.abs(z_vals - depth_max))
        # start_idx = 0
        # end_idx = -1

        fig, axs = plt.subplots(10, 1, figsize=(10, 42))
        axs[0].title.set_text('udf values, udf_min={:.8f}'.format(udf.min()))
        axs[0].plot(z_vals[start_idx:end_idx], udf[start_idx:end_idx], marker='o')
        axs[1].title.set_text('udf normal magnitude')
        axs[1].plot(z_vals[start_idx:end_idx], gradient_mag[start_idx:end_idx], marker='o')
        axs[2].title.set_text('the cosine value of ray direction and udf normal')
        axs[2].plot(z_vals[start_idx:end_idx], true_cos[start_idx:end_idx], marker='o')
        axs[3].title.set_text('weight curve, weight_sum_global={:.4f}   weight_sum_local={:.4f}'.format(
            weights[:n_samples].sum(), weights[start_idx:end_idx].sum()))
        axs[3].plot(z_vals[start_idx:end_idx], weights[start_idx:end_idx], marker='o')
        axs[3].plot([depth, depth], [0, 0.1], 'r*')
        axs[4].title.set_text('alpha curve')
        axs[4].plot(z_vals[start_idx:end_idx], alpha[start_idx:end_idx], marker='o')
        if vis_prob is not None:
            axs[5].title.set_text('vis_prob curve')
            axs[5].plot(z_vals[start_idx:end_idx], vis_prob[start_idx:end_idx], marker='o')
        axs[6].title.set_text('alpha_plus curve')
        axs[6].plot(z_vals[start_idx:end_idx], alpha_plus[start_idx:end_idx], marker='o')
        axs[7].title.set_text('alpha_minus curve')
        axs[7].plot(z_vals[start_idx:end_idx], alpha_minus[start_idx:end_idx], marker='o')

        axs[8].title.set_text('alpha_occ curve')
        axs[8].plot(z_vals[start_idx:end_idx], alpha_occ[start_idx:end_idx], marker='o')

        axs[9].title.set_text('raw_occ curve')
        axs[9].plot(z_vals[start_idx:end_idx], raw_occ[start_idx:end_idx], marker='o')
        # plt.show()

        save_dir = os.path.join(self.base_exp_dir, 'ray_statis')
        os.makedirs(os.path.join(save_dir, "color_map"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'ray_statis_step{}'.format(self.iter_step)), exist_ok=True)
        if os.path.exists(os.path.join(save_dir, "color_map")):
            cv.imwrite(os.path.join(save_dir, 'color_map',
                                    'img_px{}_py{}.png'.format(px, py)), image)
        plt.savefig(os.path.join(save_dir, 'ray_statis_step{}'.format(self.iter_step),
                                 'statis_px{}_py{}.png'.format(px, py)))
        plt.close(fig)
        np.save(os.path.join(save_dir, 'ray_statis_step{}'.format(self.iter_step),
                             'statis_px{}_py{}.npy'.format(px, py)), {'z_vals': z_vals, 'udf': udf, 'cos': true_cos})

    def validate(self, idx=-1, resolution_level=-1, only_color=False):
        # validate image
        ic(self.iter_step, idx)
        logging.info('Validate begin')
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)
        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)

        ref_c2w, src_c2ws, src_intrinsics, src_images, img_wh = self.dataset.get_ref_src_info(idx)

        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        out_rgb_pixel = []
        out_normal_fine = []
        out_depth = []

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)

            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch, rays_d_batch, near, far,
                                              color_maps=src_images,
                                              w2cs=torch.inverse(src_c2ws),
                                              intrinsics=src_intrinsics,
                                              query_c2w=ref_c2w,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            feasible = lambda key: ((key in render_out) and (render_out[key] is not None))

            # if render_out['color_coarse'] is not None:
            if feasible('color'):
                out_rgb_fine.append(render_out['color'].detach().cpu().numpy())
            if feasible('color_pixel'):
                out_rgb_pixel.append(render_out['color_pixel'].detach().cpu().numpy())
            if feasible('depth'):
                out_depth.append(render_out['depth'].detach().cpu().numpy())
            if not feasible('gradients_flip'):
                if feasible('gradients') and feasible('weights'):
                    if render_out['inside_sphere'] is not None:
                        out_normal_fine.append((render_out['gradients'] * render_out['weights'][:,
                                                                          :self.renderer.n_samples + self.renderer.n_importance,
                                                                          None] * render_out['inside_sphere'][
                                                    ..., None]).sum(dim=1).detach().cpu().numpy())
                    else:
                        out_normal_fine.append((render_out['gradients'] * render_out['weights'][:,
                                                                          :self.renderer.n_samples + self.renderer.n_importance,
                                                                          None]).sum(dim=1).detach().cpu().numpy())
            else:
                if feasible('gradients_flip') and feasible('weights'):
                    if render_out['inside_sphere'] is not None:
                        out_normal_fine.append((render_out['gradients_flip'] * render_out['weights'][:,
                                                                               :self.renderer.n_samples + self.renderer.n_importance,
                                                                               None] * render_out['inside_sphere'][
                                                    ..., None]).sum(dim=1).detach().cpu().numpy())
                    else:
                        out_normal_fine.append((render_out['gradients_flip'] * render_out['weights'][:,
                                                                               :self.renderer.n_samples + self.renderer.n_importance,
                                                                               None]).sum(dim=1).detach().cpu().numpy())
            del render_out

        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255)
        img_pixel = None
        if len(out_rgb_pixel) > 0:
            img_pixel = (np.concatenate(out_rgb_pixel, axis=0).reshape([H, W, 3]) * 256).clip(0, 255)

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy())
            normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None]).reshape([H, W, 3]) * 128 + 128).clip(0,
                                                                                                                  255)

        depth_vis = None
        if len(out_depth) > 0:
            pred_depth = (np.concatenate(out_depth, axis=0).reshape([H, W]))
            depth_vis = colorize_depth(pred_depth, near[0, 0].cpu().numpy(), far[0, 0].cpu().numpy())

        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'depth'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)

        if only_color:
            os.makedirs(os.path.join(self.base_exp_dir, 'novel_view'), exist_ok=True)
            cv.imwrite(os.path.join(self.base_exp_dir, 'novel_view',
                                    'pred_{}.png'.format(idx)), img_fine)
            cv.imwrite(os.path.join(self.base_exp_dir, 'novel_view',
                                    'gt_{}.png'.format(idx)),
                       self.dataset.image_at(idx, resolution_level=resolution_level))
            return True

        if len(out_rgb_fine) > 0:
            if len(out_rgb_pixel) > 0:
                rgbs = [img_fine, img_pixel]
            cv.imwrite(os.path.join(self.base_exp_dir, 'validations_fine',
                                    '{:0>8d}_{}.png'.format(self.iter_step, idx)),
                       np.concatenate(
                           rgbs + [self.dataset.image_at(idx, resolution_level=resolution_level)]))

        if len(out_normal_fine) > 0:
            cv.imwrite(os.path.join(self.base_exp_dir, 'normals', '{:0>8d}_{}.png'.format(self.iter_step, idx)),
                       normal_img[:, :, ::-1])

        if len(out_depth) > 0:
            cv.imwrite(os.path.join(self.base_exp_dir,
                                    'depth',
                                    '{:0>8d}_{}.png'.format(self.iter_step, idx)),
                       depth_vis[:, :, ::-1])

    def validate_novel_image(self, idx_0, idx_1, ratio, out_idx, resolution_level):
        rays_o, rays_d = self.dataset.gen_rays_between(idx_0, idx_1, ratio, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)

            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch, rays_d_batch, near, far,
                                              # alpha_inter_ratio=self.get_alpha_inter_ratio(),
                                              background_rgb=background_rgb)
            out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())

            del render_out

        img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)
        os.makedirs(os.path.join(self.base_exp_dir, 'render'), exist_ok=True)
        ic(img_fine.shape)
        print(cv.imwrite(os.path.join(self.base_exp_dir, 'render', '{}.png'.format(out_idx)), img_fine.squeeze()))
        print(os.path.join(self.base_exp_dir, 'render', '{}.png'.format(out_idx)))

    def validate_mesh(self, world_space=True, resolution=256, threshold=0.0):
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)
        vertices, triangles = self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution,
                                                             threshold=threshold, device=self.device)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(
            os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}_thresh{:.4f}_res{}.ply'.format(self.iter_step, threshold,
                                                                                              args.resolution)))

        logging.info('End')

    def extract_udf_mesh(self, world_space=False, resolution=256, dist_threshold_ratio=1.0):
        if self.model_type == 'udf':
            func = self.udf_network_fine.udf

            def func_grad(xyz):
                gradients = self.udf_network_fine.gradient(xyz)
                gradients_mag = torch.linalg.norm(gradients, ord=2, dim=-1, keepdim=True)
                gradients_norm = gradients / (gradients_mag + 1e-5)  # normalize to unit vector
                return gradients_norm

        elif self.model_type == 'neus':
            func = lambda pts: torch.abs(self.sdf_network_fine.sdf(pts))
            func_grad = self.sdf_network_fine.gradient

        try:
            pred_v, pred_f, pred_mesh, samples, indices = get_mesh_udf_fast(func, func_grad, samples=None,
                                                                            indices=None, N_MC=resolution,
                                                                            gradient=True, eps=0.005,
                                                                            border_gradients=True,
                                                                            smooth_borders=True,
                                                                            dist_threshold_ratio=dist_threshold_ratio)
        except:
            pred_v, pred_f, pred_mesh, samples, indices = get_mesh_udf_fast(func, func_grad, samples=None,
                                                                            indices=None, N_MC=resolution,
                                                                            gradient=True, eps=0.005,
                                                                            border_gradients=False,
                                                                            smooth_borders=False,
                                                                            dist_threshold_ratio=dist_threshold_ratio)

        vertices, triangles = pred_mesh.vertices, pred_mesh.faces
        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)

        os.makedirs(os.path.join(self.base_exp_dir, 'udf_meshes'), exist_ok=True)
        mesh.export(
            os.path.join(self.base_exp_dir, 'udf_meshes', 'udf_res{}_step{}.ply'.format(resolution, self.iter_step)))

    def validate_fields(self, iter_step=-1):
        os.makedirs(os.path.join(self.base_exp_dir, 'fields'), exist_ok=True)

        if iter_step < 0:
            iter_step = self.iter_step
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)

        func_d = self.sdf_network_fine.sdf if self.model_type == 'neus' else self.udf_network_fine.udf
        func_g = self.sdf_network_fine.gradient if self.model_type == 'neus' else self.udf_network_fine.gradient
        sdf = extract_fields(bound_min, bound_max, args.resolution, lambda pts: func_d(pts)[:, 0], device=self.device)
        np.save(os.path.join(self.base_exp_dir, 'fields', '{:0>8d}_dist.npy'.format(iter_step)), sdf)

        # gradients = extract_gradient_fields(bound_min, bound_max, args.resolution, lambda pts: func_g(pts)[:, 0],
        #                                     device=self.device)
        # np.save(os.path.join(self.base_exp_dir, 'fields', '{:0>8d}_gradient.npy'.format(iter_step)), gradients)

    def save_hdf5(self):
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)

        func_d = self.sdf_network_fine.sdf if self.model_type == 'neus' else self.udf_network_fine.udf
        func_g = self.sdf_network_fine.gradient if self.model_type == 'neus' else self.udf_network_fine.gradient
        sdf = extract_fields(bound_min, bound_max, args.resolution + 1, lambda pts: func_d(pts)[:, 0],
                             device=self.device)
        gradients = extract_gradient_fields(bound_min, bound_max, args.resolution + 1, lambda pts: func_g(pts)[:, 0],
                                            device=self.device)
        os.makedirs(os.path.join(self.base_exp_dir, 'hdf5'), exist_ok=True)

        out_hdf5_name = os.path.join(self.base_exp_dir, 'hdf5', 'out.hdf5')
        hdf5_file = h5py.File(out_hdf5_name, 'w')
        grid_size = args.resolution
        grid_size_1 = grid_size + 1
        hdf5_file.create_dataset(str(grid_size) + "_sdf", [grid_size_1, grid_size_1, grid_size_1], np.float32,
                                 compression=9)

        # normalize sdf
        sdf = sdf / sdf.max() * 0.5
        hdf5_file[str(grid_size) + "_sdf"][:] = sdf
        hdf5_file.close()


import matplotlib.cm


def colorize_depth(value, vmin=10, vmax=1000, cmap='plasma'):
    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.
    # squeeze last dim if it exists
    # value = value.squeeze(axis=0)

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # (nxmx4)

    img = value[:, :, :3]

    #     return img.transpose((2, 0, 1))
    return img


if __name__ == '__main__':

    # import GPUtil

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_type', type=str, default='')
    parser.add_argument('--threshold', type=float, default=0.005)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--is_finetune', default=False, action="store_true")
    parser.add_argument('--reg_weights_schedule', default=False, action="store_true",
                        help='the schedule of regularization weights')
    parser.add_argument('--vis_ray', default=False, action="store_true", help='visualize the udf of a ray for debug')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--resolution', type=int, default=128)
    parser.add_argument('--case', type=str, default='', help='the object name or index of a dataset')

    parser.add_argument('--learning_rate', type=float, default=0)
    parser.add_argument('--learning_rate_geo', type=float, default=0,
                        help='the learning rate of udf network, if do not use the global learning rate')

    parser.add_argument('--sparse_weight', type=float, default=0, help='the weight of geo regularizer')

    args = parser.parse_args()

    runner = Runner(args.conf, args.mode, args.case, args.model_type, args.is_continue, args)

    if args.mode == 'train':
        runner.train()
        runner.extract_udf_mesh(resolution=512, world_space=True, dist_threshold_ratio=5.0)
    elif args.mode == 'validate_mesh':
        runner.validate_mesh(world_space=False, resolution=args.resolution, threshold=args.threshold)
    elif args.mode == 'extract_udf_mesh':
        runner.extract_udf_mesh(resolution=args.resolution, world_space=True, dist_threshold_ratio=5.0)
    elif args.mode.startswith('validate_image'):
        for idx in [0, 10, 20, 30, 40, 50, 60, 70]:
            runner.validate(idx, resolution_level=1, only_color=True)
    elif args.mode == 'validate_fields':
        runner.validate_fields()
    elif args.mode == 'vis_one_ray':
        for i in range(1):
            # for i in range(-runner.dataset.H // 4, runner.dataset.H // 4, 1):
            print('vis_one_ray: %d' % i)
            runner.visualize_one_ray(img_idx=48, px=runner.dataset.W // 2 + i, py=runner.dataset.H // 2)

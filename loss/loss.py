import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import icecream as ic
from loss.patch_metric import SSIM, NCC
from termcolor import colored


class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, bottom):
        qn = torch.norm(bottom, p=2, dim=1).unsqueeze(dim=1) + 1e-12
        top = bottom.div(qn)

        return top


class ColorPixelLoss(nn.Module):
    def __init__(self, type='mse'):
        super(ColorPixelLoss, self).__init__()
        # if type == 'mse':
        #     self.loss_func = F.mse_loss
        # elif type == 'l1':
        self.loss_func = F.l1_loss

    def forward(self, pred, gt, mask):
        """

        :param pred: [N_pts, 3]
        :param gt: [N_pts, 3]
        :param mask: [N_pts]
        :return:
        """
        error = (pred - gt)
        # ? use mse loss or l1 loss

        if mask is not None:
            loss = self.loss_func(error, torch.zeros_like(error), reduction='sum') / (mask.sum() + 1e-4)
        else:
            loss = self.loss_func(error, torch.zeros_like(error), reduction='mean')

        return loss


class ColorPatchLoss(nn.Module):
    def __init__(self, type='ssim', h_patch_size=3):
        super(ColorPatchLoss, self).__init__()

        self.type = type  # 'l1' or 'ssim' l1 loss or SSIM loss
        self.ssim = SSIM(h_patch_size=h_patch_size)
        self.ncc = NCC(h_patch_size=h_patch_size)
        self.eps = 1e-4

        print("type {} patch_size {}".format(type, h_patch_size))

    def forward(self, pred, gt, mask, penalize_ratio=0.3):
        """

        :param pred: [N_pts, Npx, 3]
        :param gt: [N_pts, Npx, 3]
        :param weight: [N_pts]
        :param mask: [N_pts]
        :return:
        """

        if self.type == 'l1':
            error = torch.abs(pred - gt).mean(dim=-1, keepdim=False).sum(dim=-1, keepdim=False)  # [N_pts]
        elif self.type == 'ssim':
            error = self.ssim(pred[:, None, :, :], gt)[:, 0]
        elif self.type == 'ncc':
            error = 1 - self.ncc(pred[:, None, :, :], gt)[:, 0]  # ncc 1 positive, -1 negative
        elif self.type == 'ssd':
            error = ((pred - gt) ** 2).mean(dim=-1, keepdim=False).sum(dim=-1, keepdims=False)

        # todo: improve this; avoid noisy surfaces
        error = error * mask[:, 0].float()
        error, indices = torch.sort(error, descending=True)  # errors from large to small
        mask = torch.index_select(mask, 0, index=indices)
        mask[:int(penalize_ratio * mask.sum())] = False  # don't include very large errors

        return (error[mask.squeeze()]).mean()


class ColorLoss(nn.Module):
    def __init__(self, color_base_weight, color_weight, color_pixel_weight, color_patch_weight,
                 pixel_loss_type='l1', patch_loss_type='ssim', h_patch_size=3):
        super(ColorLoss, self).__init__()
        self.color_base_weight = color_base_weight
        self.color_weight = color_weight
        self.color_pixel_weight = color_pixel_weight
        self.color_patch_weight = color_patch_weight
        self.pixel_func = ColorPixelLoss(pixel_loss_type)  # should use l1 loss; mse loss will cause bluring rendering
        self.patch_func = ColorPatchLoss(patch_loss_type, h_patch_size)
        self.h_patch_size = h_patch_size

    def set_color_weights(self, color_base_weight, color_weight, color_pixel_weight, color_patch_weight):
        self.color_base_weight = color_base_weight
        self.color_weight = color_weight
        self.color_pixel_weight = color_pixel_weight
        self.color_patch_weight = color_patch_weight

    def forward(self, color_base, color, gt_color, color_pixel, pixel_mask, patch_colors, gt_patch_colors, patch_mask):
        color_base_loss = 0.0
        color_loss = 0.0
        color_pixel_loss = 0.0
        color_patch_loss = 0.0
        if color_base is not None:
            color_base_loss = self.pixel_func(color_base, gt_color, pixel_mask)
        if color is not None:
            color_loss = self.pixel_func(color, gt_color, pixel_mask)
        if color_pixel is not None:
            color_pixel_loss = self.pixel_func(color_pixel, gt_color, patch_mask)
        if patch_colors is not None:
            color_patch_loss = self.patch_func(patch_colors, gt_patch_colors, patch_mask)

        total_loss = (color_base_loss * self.color_base_weight + \
                      color_loss * self.color_weight + \
                      color_pixel_loss * self.color_pixel_weight) / (
                             self.color_base_weight + self.color_weight + self.color_pixel_weight) + \
                     color_patch_loss * self.color_patch_weight

        losses = {
            'loss': total_loss,
            'color_base_loss': color_base_loss,
            'color_loss': color_loss,
            'color_pixel_loss': color_pixel_loss,
            'color_patch_loss': color_patch_loss
        }

        return losses

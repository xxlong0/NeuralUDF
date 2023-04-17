import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.embedder import get_embedder
from termcolor import colored


# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class SDFNetwork(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False
                 # if True, the cameras are inside the scene, otherwise the cameras are outside the scene
                 ):
        super(SDFNetwork, self).__init__()

        print(colored(f"sdf network init: bias_{bias}", 'red'))

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)

    def forward(self, inputs):
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)

    def sdf(self, x):
        return self.forward(x)[:, :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)


class UDFNetwork(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 scale=1,
                 bias=0.5,
                 geometric_init=True,
                 weight_norm=True,
                 udf_type='abs',
                 ):
        super(UDFNetwork, self).__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        self.geometric_init = geometric_init

        # self.bias = 0.5
        # bias = self.bias
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                print("using geometric init")
                if l == self.num_layers - 2:

                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)

                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)
        self.relu = nn.ReLU()
        self.udf_type = udf_type

    def udf_out(self, x):
        if self.udf_type == 'abs':
            return torch.abs(x)
        elif self.udf_type == 'square':
            return x ** 2
        elif self.udf_type == 'sdf':
            return x

    def forward(self, inputs):
        inputs = inputs * self.scale
        xyz = inputs
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)

        return torch.cat([self.udf_out(x[:, :1]) / self.scale, x[:, 1:]],
                         dim=-1)

    def udf(self, x):
        return self.forward(x)[:, :1]

    def udf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.udf(x)

        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)


# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class BlendingNetwork(nn.Module):
    def __init__(self,
                 d_feature,
                 mode,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 num_ref_views,
                 num_src_views,
                 weight_norm=True,
                 multires_view=0,
                 squeeze_out=True):
        super().__init__()

        self.mode = mode
        self.squeeze_out = squeeze_out
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.global_weights = torch.nn.Parameter(torch.ones([num_ref_views, num_src_views]), requires_grad=True)

        assert d_out == num_src_views

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, points, normals, view_dirs, feature_vectors, ref_rel_idx,
                pts_pixel_color, pts_pixel_mask, pts_patch_color=None, pts_patch_mask=None):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        rendering_input = None

        normals = normals.detach()

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        global_weights = self.global_weights[ref_rel_idx:ref_rel_idx + 1]  # [1, num_src_views]
        fused_weights = global_weights + x  # [N_pts, num_src_views]

        weights_pixel = self.softmax(fused_weights)  # [n_pts, N_views]
        weights_pixel = weights_pixel * pts_pixel_mask
        weights_pixel = weights_pixel / (
                torch.sum(weights_pixel.float(), dim=1, keepdim=True) + 1e-8)  # [n_pts, N_views]

        nan_mask = torch.isnan(weights_pixel)
        if nan_mask.any():
            raise RuntimeError("NaN encountered in gumbel softmax")

        final_pixel_color = torch.sum(pts_pixel_color * weights_pixel[:, :, None], dim=1,
                                      keepdim=False)  # [N_pts, 3]

        final_pixel_mask = torch.sum(pts_pixel_mask.float(), dim=1, keepdim=True) > 0  # [N_pts, 1]

        return final_pixel_color, final_pixel_mask


class RenderingNetwork(nn.Module):
    def __init__(self,
                 d_feature,
                 mode,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 weight_norm=True,
                 multires_view=0,
                 squeeze_out=True,
                 blending_cand_views=0):
        super().__init__()

        self.mode = mode
        self.squeeze_out = squeeze_out
        self.d_out = d_out
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out + blending_cand_views]

        self.embedview_fn = None
        if multires_view > 0 and self.mode != 'no_view_dir':
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()

        self.if_blending = blending_cand_views > 0

    def forward(self, points, normals, view_dirs, feature_vectors):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        rendering_input = None
        normals = normals.detach()
        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, -1 * normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, -1 * normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        if self.squeeze_out:
            color = torch.sigmoid(x[:, :self.d_out])
        else:
            color = x[:, :self.d_out]

        if self.if_blending:
            blending_weights = x[:, self.d_out:]
            return color, blending_weights
        else:
            return color


class ResidualRenderingNetwork(nn.Module):
    def __init__(self,
                 d_feature,
                 mode,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 weight_norm=True,
                 multires_view=0,
                 squeeze_out=True,
                 blending_cand_views=10):
        super().__init__()

        self.mode = mode
        self.squeeze_out = squeeze_out
        self.d_out = d_out

        dims_base = [d_in - 3 + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]
        dims = [d_hidden + d_out + 3] + [d_hidden for _ in range(n_layers)] + [d_out + blending_cand_views]

        self.embedview_fn = None
        if multires_view > 0 and self.mode != 'no_view_dir':
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        # * base color mlps
        for l in range(0, self.num_layers - 1):
            out_dim = dims_base[l + 1]
            lin = nn.Linear(dims_base[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin_base" + str(l), lin)

        self.relu = nn.ReLU()

        self.if_blending = blending_cand_views > 0

    def forward(self, points, normals, view_dirs, feature_vectors):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        if self.mode == 'no_normal':
            rendering_input_base = torch.cat([points, feature_vectors], dim=-1)
        else:
            # ! if add normal; training will be difficult
            normals = normals.detach()
            rendering_input_base = torch.cat([points, normals, -1 * normals, feature_vectors], dim=-1)

        x = rendering_input_base

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin_base" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)
            if l == self.num_layers - 3:
                x_hidden = x

        color_base_raw = x[:, :self.d_out]
        color_base = torch.sigmoid(color_base_raw)

        rendering_input = torch.cat([view_dirs, color_base, x_hidden], dim=-1)
        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        color = torch.sigmoid(x[:, :self.d_out])

        if self.if_blending:
            blending_weights = x[:, self.d_out:]
            return color_base, color, blending_weights
        else:
            return color_base, color


def color_blend(blending_weights, img_index,
                pts_pixel_color=None,
                pts_pixel_mask=None,
                pts_patch_color=None,
                pts_patch_mask=None):
    # fuse the color of a pt by blending the interpolated colors from supporting images
    softmax = nn.Softmax(dim=-1)
    nviews = pts_pixel_color.shape[-2]
    ## extract value based on img_index
    if img_index is not None:
        x_extracted = torch.index_select(blending_weights, 1, img_index.long())
    else:
        x_extracted = blending_weights[:, :, :nviews]

    weights_pixel = softmax(x_extracted)  # [n_pts, N_views]
    weights_pixel = weights_pixel * pts_pixel_mask
    weights_pixel = weights_pixel / (
            torch.sum(weights_pixel.float(), dim=-1, keepdim=True) + 1e-8)  # [n_pts, N_views]
    final_pixel_color = torch.sum(pts_pixel_color * weights_pixel[:, :, :, None], dim=-2,
                                  keepdim=False)  # [N_pts, 3]

    final_pixel_mask = torch.sum(pts_pixel_mask.float(), dim=-1, keepdim=True) > 0  # [N_pts, 1]

    final_patch_color, final_patch_mask = None, None
    # pts_patch_color  [N_pts, N_views, Npx, 3]; pts_patch_mask  [N_pts, N_views, Npx]
    if pts_patch_color is not None:
        batch_size, n_samples, N_views, Npx, _ = pts_patch_color.shape
        patch_mask = torch.sum(pts_patch_mask, dim=-1, keepdim=False) > Npx - 1  # [batch_size, nsamples, N_views]

        weights_patch = softmax(x_extracted)  # [batch, n_samples, N_views]
        weights_patch = weights_patch * patch_mask
        weights_patch = weights_patch / (
                torch.sum(weights_patch.float(), dim=-1, keepdim=True) + 1e-8)  # [n_pts, N_views]

        final_patch_color = torch.sum(pts_patch_color * weights_patch[:, :, :, None, None], dim=-3,
                                      keepdim=False)  # [batch, nsamples, Npx, 3]
        final_patch_mask = torch.sum(patch_mask, dim=-1,
                                     keepdim=True) > 0  # [batch, nsamples, 1]  at least one image sees

    return final_pixel_color, final_pixel_mask, final_patch_color, final_patch_mask


# This implementation is borrowed from nerf-pytorch: https://github.com/yenchenlin/nerf-pytorch
class NeRF(nn.Module):
    def __init__(self,
                 D=8,
                 W=256,
                 d_in=3,
                 d_in_view=3,
                 multires=0,
                 multires_view=0,
                 output_ch=4,
                 skips=[4],
                 use_viewdirs=False,
                 occupancy=True):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.d_in = d_in
        self.d_in_view = d_in_view
        self.input_ch = 3
        self.input_ch_view = 3
        self.embed_fn = None
        self.embed_fn_view = None
        self.occupancy = occupancy

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            self.input_ch = input_ch

        if multires_view > 0:
            embed_fn_view, input_ch_view = get_embedder(multires_view, input_dims=d_in_view)
            self.embed_fn_view = embed_fn_view
            self.input_ch_view = input_ch_view

        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D - 1)])

        ### Implementation according to the official code release
        ### (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_view + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def raw2occupancy(self, raw):
        return torch.sigmoid(10 * raw)

    def forward(self, input_pts, input_views):
        if self.embed_fn is not None:
            input_pts = self.embed_fn(input_pts)
        if self.embed_fn_view is not None and input_views is not None:
            input_views = self.embed_fn_view(input_views)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)

            if input_views is None:
                return alpha

            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            return alpha, rgb
        else:
            assert False

    def gradient(self, x):
        x.requires_grad_(True)
        y = self(x, None)
        y = self.raw2occupancy(y)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)


class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val, requires_grad=True):
        super(SingleVarianceNetwork, self).__init__()
        # self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))
        self.variance = nn.Parameter(torch.Tensor([init_val]), requires_grad=requires_grad)

    def set_trainable(self):
        self.variance.requires_grad = True

    def forward(self, x):
        return torch.ones([len(x), 1]).to(x.device) * torch.exp(self.variance * 10.0)


class BetaNetwork(nn.Module):
    def __init__(self,
                 init_var_beta=0.1,
                 init_var_gamma=0.1,
                 init_var_zeta=0.05,
                 beta_min=0.00005,
                 requires_grad_beta=True,
                 requires_grad_gamma=True,
                 requires_grad_zeta=True):
        super().__init__()

        self.beta = nn.Parameter(torch.Tensor([init_var_beta]), requires_grad=requires_grad_beta)
        self.gamma = nn.Parameter(torch.Tensor([init_var_gamma]), requires_grad=requires_grad_gamma)
        self.zeta = nn.Parameter(torch.Tensor([init_var_zeta]), requires_grad=requires_grad_zeta)
        self.beta_min = beta_min

    def get_beta(self):
        return torch.exp(self.beta * 10).clip(0, 1./self.beta_min)

    def get_gamma(self):
        return torch.exp(self.gamma * 10)

    def get_zeta(self):
        """
        used for udf2prob mapping zeta*x/(1+zeta*x)
        :return:
        :rtype:
        """
        return self.zeta.abs()

    def set_beta_trainable(self):
        self.beta.requires_grad = True

    @torch.no_grad()
    def set_gamma(self, x):
        self.gamma = nn.Parameter(torch.Tensor([x]),
                                  requires_grad=self.gamma.requires_grad).to(self.gamma.device)

    def forward(self):
        beta = self.get_beta()
        gamma = self.get_gamma()
        zeta = self.get_zeta()
        return beta, gamma, zeta

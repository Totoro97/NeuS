import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.embedder import get_embedder

import logging

# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class SDFNetwork(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 n_scenes,
                 scenewise_split_type='interleave',
                 skip_in=(4,),
                 multires=0,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False):
        """
        n_scenes
            int
            Spawn `n_scenes` copies of every other layer, each trained independently. During the
            forward pass, take a scene index `i` and use the `i`th copy at each such layer.

        scenewise_split_type
            str
            One of:
            - 'interleave'
            - 'interleave_with_skips'
            - 'append_half'
            - 'prepend_half'
            - 'replace_last_half'
            - 'replace_first_half'
        """
        super().__init__()

        self.scenewise_split_type = scenewise_split_type
        if scenewise_split_type in ('append_half', 'prepend_half'):
            num_scene_specific_layers = (n_layers + 1) // 2
            n_layers += num_scene_specific_layers
        elif scenewise_split_type in ('replace_last_half', 'replace_first_half'):
            num_scene_specific_layers = (n_layers + 1) // 2

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims) - 1
        self.skip_in = skip_in
        self.scale = scale

        self.linear_layers = nn.ModuleList()
        total_scene_specific_layers = 0

        for l in range(self.num_layers):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            def create_linear_layer():
                lin = nn.Linear(dims[l], out_dim)

                if geometric_init:
                    if l == self.num_layers - 1:
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

                return lin

            if scenewise_split_type == 'interleave':
                layer_is_scene_specific = l % 2 == 1
            elif scenewise_split_type == 'interleave_with_skips':
                layer_is_scene_specific = l % 2 == 1 and dims[l] == out_dim
            elif scenewise_split_type in ('append_half', 'replace_last_half'):
                layer_is_scene_specific = l >= self.num_layers - num_scene_specific_layers
            elif scenewise_split_type in ('prepend_half', 'replace_first_half'):
                layer_is_scene_specific = l < num_scene_specific_layers
            else:
                raise ValueError(
                    f"Wrong value for `scenewise_split_type`: '{scenewise_split_type}'")

            if layer_is_scene_specific:
                lin = nn.ModuleList([create_linear_layer() for _ in range(n_scenes)])
                total_scene_specific_layers += 1
            else:
                lin = create_linear_layer()

            self.linear_layers.append(lin)

        logging.info(
            f"SDF network got {total_scene_specific_layers} (out of " \
            f"{self.num_layers}) scene-specific layers")

        self.activation = nn.Softplus(beta=100)

    def forward(self, inputs, scene_idx):
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        for l in range(self.num_layers):
            lin = self.linear_layers[l]
            layer_is_scene_specific = type(lin) is torch.nn.ModuleList

            if layer_is_scene_specific:
                lin = lin[scene_idx]
                if self.scenewise_split_type == 'interleave_with_skips':
                    skip_connection = x

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if layer_is_scene_specific and self.scenewise_split_type == 'interleave_with_skips':
                x += skip_connection

            if l < self.num_layers - 1:
                x = self.activation(x)

        return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)

    def sdf(self, x, scene_idx):
        return self(x, scene_idx)[:, :1]

    def gradient(self, x, scene_idx):
        with torch.enable_grad():
            x.requires_grad_(True)
            y = self.sdf(x, scene_idx)
            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            gradients = torch.autograd.grad(
                outputs=y,
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
            return gradients.unsqueeze(1)

    def switch_to_finetuning(self, algorithm='pick'):
        """
        Switch the network trained on multiple scenes to the 'finetuning mode',
        to finetune it to some new (one) scene.

        algorithm
            str
            One of:
            - pick (take the 0th scene's 'subnetwork')
        """
        if algorithm == 'pick':
            for i in range(self.num_layers):
                if type(self.linear_layers[i]) is nn.ModuleList:
                    self.linear_layers[i] = nn.ModuleList([self.linear_layers[i][0]])
        else:
            raise ValueError(f"Unknown algorithm: '{algorithm}'")

    def parameters(self, scenewise_layers_only=False):
        def is_scene_specific(name: str):
            name = name.split('.')

            try:
                return name[0] == 'linear_layers' and name[1].isdigit() and name[2].isdigit()
            except IndexError:
                return False

        if scenewise_layers_only:
            return (x for name, x in super().named_parameters() if is_scene_specific(name))
        else:
            return super().parameters()


# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
# TODO: remove repetitive code from SDFNetwork
class RenderingNetwork(nn.Module):
    def __init__(self,
                 d_feature,
                 mode,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 n_scenes,
                 scenewise_split_type='interleave',
                 weight_norm=True,
                 multires_view=0,
                 squeeze_out=True):
        super().__init__()

        self.mode = mode
        self.squeeze_out = squeeze_out

        self.scenewise_split_type = scenewise_split_type
        if scenewise_split_type in ('append_half', 'prepend_half'):
            num_scene_specific_layers = (n_layers + 1) // 2
            n_layers += num_scene_specific_layers
        elif scenewise_split_type in ('replace_last_half', 'replace_first_half'):
            num_scene_specific_layers = (n_layers + 1) // 2

        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims) - 1

        self.linear_layers = nn.ModuleList()
        total_scene_specific_layers = 0

        for l in range(0, self.num_layers):
            out_dim = dims[l + 1]

            def create_linear_layer():
                lin = nn.Linear(dims[l], out_dim)

                if weight_norm:
                    lin = nn.utils.weight_norm(lin)

                return lin

            if scenewise_split_type == 'interleave':
                layer_is_scene_specific = l % 2 == 1
            elif scenewise_split_type == 'interleave_with_skips':
                layer_is_scene_specific = l % 2 == 1 and dims[l] == out_dim
            elif scenewise_split_type in ('append_half', 'replace_last_half'):
                layer_is_scene_specific = l >= self.num_layers - num_scene_specific_layers
            elif scenewise_split_type in ('prepend_half', 'replace_first_half'):
                layer_is_scene_specific = l < num_scene_specific_layers
            else:
                raise ValueError(
                    f"Wrong value for `scenewise_split_type`: '{scenewise_split_type}'")

            if layer_is_scene_specific:
                lin = nn.ModuleList([create_linear_layer() for _ in range(n_scenes)])
                total_scene_specific_layers += 1
            else:
                lin = create_linear_layer()

            self.linear_layers.append(lin)

        logging.info(
            f"Rendering network got {total_scene_specific_layers} (out of " \
            f"{self.num_layers}) scene-specific layers")

        self.relu = nn.ReLU()

    def forward(self, points, normals, view_dirs, feature_vectors, scene_idx):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        rendering_input = None

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers):
            lin = self.linear_layers[l]
            layer_is_scene_specific = type(lin) is torch.nn.ModuleList

            if layer_is_scene_specific:
                lin = lin[scene_idx]
                if self.scenewise_split_type == 'interleave_with_skips':
                    skip_connection = x

            x = lin(x)

            if layer_is_scene_specific and self.scenewise_split_type == 'interleave_with_skips':
                x += skip_connection

            if l < self.num_layers - 1:
                x = self.relu(x)

        if self.squeeze_out:
            x = torch.sigmoid(x)
        return x

    def switch_to_finetuning(self, algorithm='pick'):
        """
        Switch the network trained on multiple scenes to the 'finetuning mode',
        to finetune it to some new (one) scene.

        algorithm
            str
            One of:
            - pick (take the 0th scene's 'subnetwork')
        """
        if algorithm == 'pick':
            for i in range(self.num_layers):
                if type(self.linear_layers[i]) is nn.ModuleList:
                    self.linear_layers[i] = nn.ModuleList([self.linear_layers[i][0]])
        else:
            raise ValueError(f"Unknown algorithm: '{algorithm}'")

    def parameters(self, scenewise_layers_only=False):
        def is_scene_specific(name: str):
            name = name.split('.')

            try:
                return name[0] == 'linear_layers' and name[1].isdigit() and name[2].isdigit()
            except IndexError:
                return False

        if scenewise_layers_only:
            return (x for name, x in super().named_parameters() if is_scene_specific(name))
        else:
            return super().parameters()

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
                 use_viewdirs=False):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.d_in = d_in
        self.d_in_view = d_in_view
        self.input_ch = 3
        self.input_ch_view = 3
        self.embed_fn = None
        self.embed_fn_view = None

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

    def forward(self, input_pts, input_views):
        if self.embed_fn is not None:
            input_pts = self.embed_fn(input_pts)
        if self.embed_fn_view is not None:
            input_views = self.embed_fn_view(input_views)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            return alpha, rgb
        else:
            assert False

class MultiSceneNeRF(nn.ModuleList):
    def __init__(self, n_scenes, *args, **kwargs):
        super().__init__([NeRF(*args, **kwargs) for _ in range(n_scenes)])

    def switch_to_finetuning(self, algorithm='pick'):
        """
        Switch the network trained on multiple scenes to the 'finetuning mode',
        to finetune it to some new (one) scene.

        algorithm
            str
            One of:
            - pick (take the 0th scene's 'subnetwork')
        """
        if algorithm == 'pick':
            super().__init__([self[0]])
        else:
            raise ValueError(f"Unknown algorithm: '{algorithm}'")

    def parameters(self, scenewise_layers_only=False):
        return super().parameters()

class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1], device=x.device) * torch.exp(self.variance * 10.0)

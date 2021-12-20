from models.dataset import Dataset
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, MultiSceneNeRF
from models.renderer import NeuSRenderer

import cv2
import numpy as np
import cv2 as cv
import trimesh
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from icecream import ic
from tqdm import tqdm
from pyhocon import ConfigFactory, ConfigTree

import random
import pathlib
import os
import time
import logging
import argparse
from shutil import copyfile


def psnr(color_fine, true_rgb, mask):
    assert mask.shape[:-1] == color_fine.shape[:-1] and mask.shape[-1] == 1
    return 20.0 * torch.log10(
        1.0 / (((color_fine - true_rgb)**2 * mask).sum() / (mask.sum() * 3.0 + 1e-5)).sqrt())


class Runner:
    def __init__(self,
        conf_path: pathlib.Path, checkpoint_path: pathlib.Path = None,
        extra_config_args: str = None, mode: str = 'train'):

        self.device = torch.device('cuda')

        assert conf_path or checkpoint_path, "Specify at least config or checkpoint"

        # The eventual configuration, gradually filled from various sources
        self.conf = ConfigFactory.parse_string("")

        def update_config_tree(target: ConfigTree, source: ConfigTree, current_prefix: str = ''):
            """
            Recursively update values in `target` with those in `source`.

            current_prefix:
                str
                No effect, only used for logging.
            """
            for key in source.keys():
                if key not in target:
                    target[key] = source[key]
                else:
                    assert type(source[key]) == type(target[key]), \
                        f"Types differ in ConfigTrees: asked to update '{type(target[key])}' " \
                        f"with '{type(source[key])}' at key '{current_prefix}{key}'"

                    if type(source[key]) is ConfigTree:
                        update_config_tree(target[key], source[key], f'{current_prefix}{key}.')
                    else:
                        if target[key] != source[key]:
                            logging.info(
                                f"Updating config value at '{current_prefix}{key}'. "
                                f"Old: '{target[key]}', new: '{source[key]}'")

                        target[key] = source[key]

        # Load the config file, for now just to extract the checkpoint path from it
        self.conf_path = conf_path
        if conf_path is not None:
            logging.info(f"Using config '{conf_path}'")
            config_from_file = ConfigFactory.parse_file(conf_path)

            checkpoint_path_from_config = \
                config_from_file.get_string('train.checkpoint_path', default=None)

            if checkpoint_path is None:
                checkpoint_path = checkpoint_path_from_config

        # Try to find the latest checkpoint
        if checkpoint_path is None:
            checkpoints_dir = pathlib.Path(self.base_exp_dir) / "checkpoints"
            if checkpoints_dir.is_dir():
                checkpoints = sorted(checkpoints_dir.iterdir())
            else:
                checkpoints = []

            if checkpoints:
                checkpoint_path = checkpoints[-1]

        # Load the checkpoint, for now just to extract config from there
        if checkpoint_path is not None:
            logging.info(f"Loading checkpoint '{checkpoint_path}'")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            if 'config' in checkpoint:
                # Temporary dynamic defaults for backward compatibility. TODO: remove
                if 'dataset.original_num_scenes' not in checkpoint['config']:
                    checkpoint['config']['dataset']['original_num_scenes'] = \
                        len(checkpoint['config']['dataset.data_dirs'])

                update_config_tree(self.conf, checkpoint['config'])

        # Now actually process the config file and merge it
        if conf_path is not None:
            update_config_tree(self.conf, config_from_file)

        # Finally, update config with extra command line args
        if extra_config_args is not None:
            update_config_tree(self.conf, ConfigFactory.parse_string(extra_config_args))

        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.dataset = Dataset(self.conf['dataset'])

        logging.info(f"Experiment dir: {self.base_exp_dir}")

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        # List of (scene_idx, image_idx) pairs. Example: [[0, 4], [1, 2]].
        # -1 for random. Examples: [-1] or [[0, 4], -1]
        self.val_images_idxs = self.conf.get_list('train.val_images_idxs')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.base_learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.learning_rate_reduce_steps = \
            [int(x) for x in self.conf.get_list('train.learning_rate_reduce_steps')]
        self.learning_rate_reduce_factor = self.conf.get_float('train.learning_rate_reduce_factor')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)
        self.restart_from_iter = self.conf.get_int('train.restart_from_iter', default=None)
        self.iter_step = None if self.restart_from_iter is None else self.restart_from_iter

        if 'train.restart_from_iter' in self.conf:
            del self.conf['train']['restart_from_iter'] # for proper checkpoint auto-restarts

        self.finetune = self.conf.get_bool('train.finetune', default=False)
        self.train_scenewise_layers_only = \
            self.conf.get_bool('train.train_scenewise_layers_only', default=False)
        load_optimizer = \
            self.conf.get_bool('train.load_optimizer', default=not self.finetune)

        if self.finetune:
            assert self.dataset.num_scenes == 1, "Can only finetune to one scene"

        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.mode = mode
        self.model_list = []
        self.writer = None

        # Networks
        current_num_scenes = self.conf.get_int('dataset.original_num_scenes', default=self.dataset.num_scenes)
        self.nerf_outside = MultiSceneNeRF(**self.conf['model.nerf'], n_scenes=current_num_scenes).to(self.device)
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network'], n_scenes=current_num_scenes).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network'], n_scenes=current_num_scenes).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)

        def get_optimizer(scenewise_layers_only=False):
            params_to_train = []
            params_to_train += list(self.nerf_outside.parameters(scenewise_layers_only))
            params_to_train += list(self.sdf_network.parameters(scenewise_layers_only))
            params_to_train += list(self.deviation_network.parameters())
            params_to_train += list(self.color_network.parameters(scenewise_layers_only))
            logging.info(f"Got {len(params_to_train)} trainable tensors")

            return torch.optim.Adam(params_to_train, lr=1e10)

        self.renderer = NeuSRenderer(self.nerf_outside,
                                     self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                     **self.conf['model.neus_renderer'])

        if load_optimizer:
            self.optimizer = get_optimizer()

        self.load_checkpoint(checkpoint, load_optimizer)

        if self.finetune:
            self.sdf_network.switch_to_finetuning()
            self.color_network.switch_to_finetuning()
            self.nerf_outside.switch_to_finetuning()

        if not load_optimizer:
            self.optimizer = get_optimizer(self.train_scenewise_layers_only)

        # In case of finetuning
        self.conf['dataset']['original_num_scenes'] = self.dataset.num_scenes

        # Backup codes and configs for debug
        if self.mode[:5] == 'train':
            self.file_backup()

    def train(self):
        self.writer = SummaryWriter(log_dir=self.base_exp_dir)
        res_step = self.end_iter - self.iter_step

        data_loader = iter(self.dataset.get_dataloader())

        for iter_i in tqdm(range(res_step)):
            start_time = time.time()

            scene_idx, (rays_o, rays_d, true_rgb, mask, near, far) = next(data_loader)

            rays_o = rays_o.cuda()
            rays_d = rays_d.cuda()
            true_rgb = true_rgb.cuda()
            mask = mask.cuda()
            near = near.cuda()
            far = far.cuda()

            background_rgb = None
            if self.use_white_bkgd:
                background_rgb = torch.ones([1, 3])

            if self.mask_weight > 0.0:
                mask = (mask > 0.5).float()
            else:
                mask = torch.ones_like(mask)

            mask_sum = mask.sum() + 1e-5
            render_out = self.renderer.render(rays_o, rays_d, near, far, scene_idx,
                                              background_rgb=background_rgb,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio())

            color_fine = render_out['color_fine']
            s_val = render_out['s_val']
            cdf_fine = render_out['cdf_fine']
            gradient_error = render_out['gradient_error']
            weight_max = render_out['weight_max']
            weight_sum = render_out['weight_sum']

            # Loss
            color_error = (color_fine - true_rgb) * mask
            color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
            psnr_train = psnr(color_fine, true_rgb, mask)

            eikonal_loss = gradient_error

            mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)

            loss = color_fine_loss +\
                   eikonal_loss * self.igr_weight +\
                   mask_loss * self.mask_weight

            learning_rate = self.update_learning_rate() # the value is only needed for logging
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            step_time = time.time() - start_time

            with torch.no_grad():
                self.iter_step += 1

                self.writer.add_scalar('Loss/Total', loss, self.iter_step)
                self.writer.add_scalar('Loss/L1', color_fine_loss, self.iter_step)
                self.writer.add_scalar('Loss/Eikonal', eikonal_loss, self.iter_step)
                self.writer.add_scalar('Loss/PSNR (train)', psnr_train, self.iter_step)
                self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)
                self.writer.add_scalar('Statistics/Learning rate', learning_rate, self.iter_step)
                self.writer.add_scalar('Statistics/cdf', (cdf_fine[:, :1] * mask).sum() / mask_sum, self.iter_step)
                self.writer.add_scalar('Statistics/weight_max', (weight_max * mask).sum() / mask_sum, self.iter_step)
                self.writer.add_scalar('Statistics/Step time', step_time, self.iter_step)

                if self.iter_step % self.report_freq == 0:
                    print(self.base_exp_dir)
                    print('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, loss, self.optimizer.param_groups[0]['lr']))

                if self.iter_step % self.save_freq == 0 or self.iter_step == self.end_iter or self.iter_step == 1:
                    self.save_checkpoint()

                if self.iter_step % self.val_freq == 0: # or self.iter_step == 1:
                    self.validate_images(self.val_images_idxs)

                if self.iter_step % self.val_mesh_freq == 0 or self.iter_step == self.end_iter:
                    self.validate_mesh()

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        learning_rate = self.base_learning_rate * learning_factor

        for reduce_step in self.learning_rate_reduce_steps:
            if self.iter_step >= reduce_step:
                learning_rate *= self.learning_rate_reduce_factor

        for g in self.optimizer.param_groups:
            g['lr'] = learning_rate

        return learning_rate

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint: dict, load_optimizer: bool = True):
        def load_weights(module, state_dict):
            # backward compatibility:
            # replace 'lin(0-9)' (old convention) with 'linear_layers.' (new convention)
            for tensor_name in list(state_dict.keys()):
                if tensor_name.startswith('lin') and tensor_name[3].isdigit():
                    new_tensor_name = f'linear_layers.{tensor_name[3:]}'
                    state_dict[new_tensor_name] = state_dict[tensor_name]
                    del state_dict[tensor_name]

            module.load_state_dict(state_dict)
            # missing_keys, unexpected_keys = module.load_state_dict(state_dict, strict=False)

            # if missing_keys:
            #     raise RuntimeError(
            #         f"Missing keys in checkpoint: {missing_keys}\n" \
            #         f"Unexpected keys: {unexpected_keys}")
            # if unexpected_keys:
            #     logging.warning(f"Ignoring unexpected keys in checkpoint: {unexpected_keys}")

        load_weights(self.nerf_outside, checkpoint['nerf'])
        load_weights(self.sdf_network, checkpoint['sdf_network_fine'])
        load_weights(self.deviation_network, checkpoint['variance_network_fine'])
        load_weights(self.color_network, checkpoint['color_network_fine'])
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        if self.iter_step is None:
            self.iter_step = checkpoint['iter_step']

        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
            'nerf': self.nerf_outside.state_dict(),
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
            'config': self.conf,
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', f'ckpt_{self.iter_step:07d}.pth'))

    def validate_images(self, idxs=[-1], resolution_level=-1):
        def get_random_dataset_idx():
            scene_idx = random.randint(0, self.dataset.num_scenes - 1)
            image_idx = random.randint(0, len(self.dataset.images[scene_idx]) - 1)
            return scene_idx, image_idx

        # Treat a plain integer as a camera index in the scene #0
        idxs = [(0, idx) if type(idx) is int else idx for idx in idxs]
        # Treat -1s as "pick a random dataset camera"
        idxs = [get_random_dataset_idx() if idx[0] < 0 else idx for idx in idxs]

        print('Validate: iter: {}, (scene, camera): {}'.format(self.iter_step, idxs))

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level

        render_images = []
        normal_images = []
        psnr_val = []

        for val_scene_idx, val_image_idx in tqdm(idxs):
            rays_o, rays_d, true_rgb, mask, near, far = self.dataset.gen_rays_at(
                val_scene_idx, val_image_idx, resolution_level=resolution_level)

            H, W, _ = rays_o.shape
            rays_o = rays_o.cuda().reshape(-1, 3).split(self.batch_size)
            rays_d = rays_d.cuda().reshape(-1, 3).split(self.batch_size)
            near = near.cuda().reshape(-1, 1).split(self.batch_size)
            far = far.cuda().reshape(-1, 1).split(self.batch_size)

            out_rgb_fine = []
            out_normal_fine = []

            for rays_o_batch, rays_d_batch, near_batch, far_batch in zip(rays_o, rays_d, near, far):
                background_rgb = \
                        torch.ones([1, 3], device=rays_o.device) if self.use_white_bkgd else None

                render_out = self.renderer.render(rays_o_batch,
                                                  rays_d_batch,
                                                  near_batch,
                                                  far_batch,
                                                  val_scene_idx,
                                                  cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                                  background_rgb=background_rgb)

                def feasible(key): return (key in render_out) and (render_out[key] is not None)

                if feasible('color_fine'):
                    out_rgb_fine.append(render_out['color_fine'].cpu())
                if feasible('gradients') and feasible('weights'):
                    n_samples = self.renderer.n_samples + self.renderer.n_importance
                    normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                    if feasible('inside_sphere'):
                        normals = normals * render_out['inside_sphere'][..., None]
                    normals = normals.sum(dim=1).cpu()
                    out_normal_fine.append(normals)
                del render_out

            img_fine = torch.cat(out_rgb_fine).reshape(H, W, 3).clamp(0.0, 1.0)

            normal_img = torch.cat(out_normal_fine)
            rot = torch.inverse(self.dataset.pose_all[val_scene_idx][val_image_idx, :3, :3].cpu())
            normal_img = ((rot[None, :, :] @ normal_img[:, :, None]).reshape(H, W, 3) * 0.5 + 0.5)
            normal_img = normal_img.clamp(0.0, 1.0)

            # os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
            # os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)

            psnr_val.append(psnr(img_fine, true_rgb, mask))

            render_image = torch.cat((img_fine, true_rgb), dim=1)
            normal_image = normal_img

            render_images.append(render_image)
            normal_images.append(normal_image)

        render_images = cv2.cvtColor(torch.cat(render_images).numpy(), cv2.COLOR_BGR2RGB)
        normal_images = cv2.cvtColor(torch.cat(normal_images).numpy(), cv2.COLOR_BGR2RGB)

        self.writer.add_image(
            'Image/Render (val)', render_images, self.iter_step, dataformats='HWC')
        self.writer.add_image(
            'Image/Normals (val)', normal_images, self.iter_step, dataformats='HWC')
        self.writer.add_scalar('Loss/PSNR (val)', np.mean(psnr_val), self.iter_step)

    def render_novel_image(self, scene_idx, idx_0, idx_1, ratio, resolution_level):
        """
        Interpolate view between two cameras.
        """
        rays_o, rays_d, near, far = self.dataset.gen_rays_between(
            scene_idx, idx_0, idx_1, ratio, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.cuda().reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.cuda().reshape(-1, 3).split(self.batch_size)
        near = near.cuda().reshape(-1, 1).split(self.batch_size)
        far = far.cuda().reshape(-1, 1).split(self.batch_size)

        out_rgb_fine = []
        for rays_o_batch, rays_d_batch, near_batch, far_batch in zip(rays_o, rays_d, near, far):
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near_batch,
                                              far_batch,
                                              scene_idx,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())

            del render_out

        img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 255).clip(0, 255).astype(np.uint8)
        return img_fine

    def validate_mesh(self, scene_idx=0, world_space=False, resolution=64, threshold=0.0):
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)

        vertices, triangles =self.renderer.extract_geometry(
            scene_idx, bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))

        logging.info('End')

    def interpolate_view(self, scene_idx, img_idx_0, img_idx_1):
        images = []
        n_frames = 30
        for i in tqdm(range(n_frames)):
            images.append(self.render_novel_image(
                scene_idx, img_idx_0, img_idx_1,
                np.sin(((i / n_frames) - 0.5) * np.pi) * 0.5 + 0.5, resolution_level=4))
        for i in range(n_frames):
            images.append(images[n_frames - i - 1])

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        video_dir = os.path.join(self.base_exp_dir, 'render')
        os.makedirs(video_dir, exist_ok=True)
        h, w, _ = images[0].shape
        writer = cv.VideoWriter(
            os.path.join(
                video_dir,
                f"{self.iter_step:0>8d}_scene{scene_idx:03d}_{img_idx_0}_{img_idx_1}.mp4"),
            fourcc, 15, (w, h))

        for image in images:
            writer.write(image)

        writer.release()


if __name__ == '__main__':
    print('Hello Wooden')

    # torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=pathlib.Path, default=None)
    parser.add_argument('--checkpoint_path', type=pathlib.Path, default=None)
    parser.add_argument('--extra_config_args', type=str, default=None)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    runner = Runner(args.conf, args.checkpoint_path, args.extra_config_args, args.mode)

    if args.mode == 'train':
        runner.train()
    elif args.mode == 'validate_mesh':
        runner.validate_mesh(world_space=True, resolution=512, threshold=args.mcube_threshold)
    elif args.mode.startswith('interpolate_'):
        # Interpolate views given [optional: scene index and] two image indices
        arguments = args.mode.split('_')[1:]
        if len(arguments) == 2:
            scene_idx, (img_idx_0, img_idx_1) = 0, map(int, arguments)
        elif len(arguments) == 3:
            scene_idx, img_idx_0, img_idx_1 = map(int, arguments)
        else:
            raise ValueError(f"Wrong number of '_' arguments (must be 3 or 4): {args.mode}")

        runner.interpolate_view(scene_idx, img_idx_0, img_idx_1)
    else:
        raise ValueError(f"Wrong '--mode': {args.mode}")

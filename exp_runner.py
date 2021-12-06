from models.dataset import Dataset
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF
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
from pyhocon import ConfigFactory

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

def load_config(conf_path):
    with open(conf_path) as f:
        return ConfigFactory.parse_string(f.read())


class Runner:
    def __init__(self, conf_path, mode='train'):
        self.device = torch.device('cuda')

        # Configuration
        self.conf_path = conf_path
        self.conf = load_config(conf_path)

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
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)
        self.restart_from_iter = self.conf.get_int('train.restart_from_iter', default=None)
        self.iter_step = None if self.restart_from_iter is None else self.restart_from_iter

        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.checkpoint_path = self.conf.get_string('train.checkpoint_path', default=None)
        self.mode = mode
        self.model_list = []
        self.writer = None

        # Networks
        params_to_train = []
        self.nerf_outside = torch.nn.ModuleList([NeRF(**self.conf['model.nerf']).to(self.device) for _ in range(self.dataset.num_scenes)])
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network'], n_scenes=self.dataset.num_scenes).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network'], n_scenes=self.dataset.num_scenes).to(self.device)
        params_to_train += list(self.nerf_outside.parameters())
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())

        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

        self.renderer = NeuSRenderer(self.nerf_outside,
                                     self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                     **self.conf['model.neus_renderer'])

        # Try to find the latest checkpoint
        if self.checkpoint_path is None:
            checkpoints_dir = pathlib.Path(self.base_exp_dir) / "checkpoints"
            if checkpoints_dir.is_dir():
                checkpoints = sorted(checkpoints_dir.iterdir())
            else:
                checkpoints = []

            if checkpoints:
                self.checkpoint_path = checkpoints[-1]

        # Load checkpoint
        if self.checkpoint_path is None:
            self.iter_step = self.iter_step or 0
        else:
            logging.info(f"Loading from checkpoint {self.checkpoint_path}")
            self.load_checkpoint(self.checkpoint_path)

        # Backup codes and configs for debug
        if self.mode[:5] == 'train':
            self.file_backup()

    def train(self):
        self.writer = SummaryWriter(log_dir=self.base_exp_dir)
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        data_idxs = self.get_dataset_indices()

        for iter_i in tqdm(range(res_step)):
            # Shuffle every so often
            if self.iter_step % len(data_idxs) == 0:
                random.shuffle(data_idxs)

            scene_idx, image_idx = data_idxs[self.iter_step % len(data_idxs)]
            data = self.dataset.gen_random_rays_at(scene_idx, image_idx, self.batch_size)

            rays_o, rays_d, true_rgb, mask = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10]
            near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)

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

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                self.iter_step += 1

                self.writer.add_scalar('Loss/Total', loss, self.iter_step)
                self.writer.add_scalar('Loss/L1', color_fine_loss, self.iter_step)
                self.writer.add_scalar('Loss/Eikonal', eikonal_loss, self.iter_step)
                self.writer.add_scalar('Loss/PSNR (train)', psnr_train, self.iter_step)
                self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)
                self.writer.add_scalar('Statistics/cdf', (cdf_fine[:, :1] * mask).sum() / mask_sum, self.iter_step)
                self.writer.add_scalar('Statistics/weight_max', (weight_max * mask).sum() / mask_sum, self.iter_step)

                if self.iter_step % self.report_freq == 0:
                    print(self.base_exp_dir)
                    print('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, loss, self.optimizer.param_groups[0]['lr']))

                if self.iter_step % self.save_freq == 0 or self.iter_step == self.end_iter:
                    self.save_checkpoint()

                if self.iter_step % self.val_freq == 0 or self.iter_step == 1:
                    self.validate_images(self.val_images_idxs)

                if self.iter_step % self.val_mesh_freq == 0 or self.iter_step == self.end_iter:
                    self.validate_mesh()

                self.update_learning_rate()

    def get_dataset_indices(self):
        # Generate all possible pairs (scene_idx, image_idx)
        num_images = list(map(len, self.dataset.images))
        all_data_idxs = [(scene_idx, image_idx) \
            for scene_idx in range(self.dataset.num_scenes) \
            for image_idx in range(num_images[scene_idx])]

        return all_data_idxs

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

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

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

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
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
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

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
            rays_o, rays_d, true_rgb, mask = self.dataset.gen_rays_at(
                val_scene_idx, val_image_idx, resolution_level=resolution_level)

            true_rgb = true_rgb.cpu()
            mask = mask.cpu()

            H, W, _ = rays_o.shape
            rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
            rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

            out_rgb_fine = []
            out_normal_fine = []

            for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
                near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
                background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

                render_out = self.renderer.render(rays_o_batch,
                                                  rays_d_batch,
                                                  near,
                                                  far,
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
        rays_o, rays_d = self.dataset.gen_rays_between(
            scene_idx, idx_0, idx_1, ratio, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
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
        writer = cv.VideoWriter(os.path.join(video_dir,
                                             '{:0>8d}_{}_{}.mp4'.format(self.iter_step, img_idx_0, img_idx_1)),
                                fourcc, 15, (w, h))

        for image in images:
            writer.write(image)

        writer.release()


if __name__ == '__main__':
    print('Hello Wooden')

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    runner = Runner(args.conf, args.mode)

    if args.mode == 'train':
        runner.train()
    elif args.mode == 'validate_mesh':
        runner.validate_mesh(world_space=True, resolution=512, threshold=args.mcube_threshold)
    elif args.mode.startswith('interpolate_'):
        # Interpolate views given [optional: scene index and] two image indices
        arguments = args.mode.split('_')[1:]
        if len(arguments) == 3:
            img_idx_0, img_idx_1 = map(int, arguments)
        elif len(arguments) == 4:
            scene_idx, img_idx_0, img_idx_1 = map(int, arguments)
        else:
            raise ValueError(f"Wrong number of '_' arguments (must be 3 or 4): {args.mode}")

        runner.interpolate_view(scene_idx, img_idx_0, img_idx_1)

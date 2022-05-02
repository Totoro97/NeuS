import os
import pdb
import time
import logging
import argparse
import numpy as np
import cv2 as cv
import trimesh
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from icecream import ic
from tqdm import tqdm
from pyhocon import ConfigFactory
from models.dataset import Dataset
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF
from models.renderer import NeuSRenderer
from copy import deepcopy
from torch import multiprocessing as mp
import queue


class MetaWeights:
    """Class to consolidate."""
    def __init__(self, conf):
        self.conf = conf
        # Initial weights are random
        self.nerf_sd = NeRF(**self.conf['model.nerf']).state_dict()
        self.sdf_sd = SDFNetwork(**self.conf['model.sdf_network']).state_dict()
        self.deviation_sd = SingleVarianceNetwork(**self.conf['model.variance_network']).state_dict()
        self.color_sd = RenderingNetwork(**self.conf['model.rendering_network']).state_dict()

        self.epsilon = conf['meta.epsilon']
        self.base_exp_dir = conf['meta.base_exp_dir']
        self.iter_step = 0
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))

    def get_state_dicts(self):
        """This dict of state_dicts is passed to each process."""
        return {
            'nerf': self.nerf_sd,
            'sdf': self.sdf_sd,
            'deviation': self.deviation_sd,
            'color': self.color_sd
        }

    @torch.no_grad()
    def update(self, sd_infos: list):
        """
        Allows batch updates.

        sd_infos is a list of dictionaries of state_dicts.
        Each entry of sd_infos should be a dictionary returned by
        runner:get_state_dicts() (same format as this classes get_state_dicts().

        The runner:get_state_dicts() returns JUST the weights of the
        network at the end of training.
        """
        # Accumulate render weights
        for network in ["nerf", "sdf", "deviation", "color"]:
            avg_weights = self.average_dicts([info[network] for info in sd_infos])
            neg_grad = self.subtract_dicts(avg_weights, getattr(self, network + "_sd"))
            # TODO: we could apply a fancier optimizer with neg_grad
            # TODO: instead we just add it, like SGD
            # TODO: this had the best performance in the meta-nerf paper
            new_sd = self.add_dicts_w_epsilon(getattr(self, network + "_sd"), neg_grad)
            setattr(self, "network" + "_sd", new_sd)

        self.iter_step += 1

    def average_dicts(self, ds: list):
        """Given nn.state_dicts on the same device, average them.

        NOTE: replaces the first dicts info with the average.
        """
        for d_idx in range(1, len(ds)):
            for key in ds[0]:
                with torch.no_grad():
                    ds[0][key] += ds[d_idx][key]

        for key in ds[0]:
            with torch.no_grad():
                ds[0][key] = (ds[0][key] / len(ds)).to(ds[0][key].dtype)
        return ds[0]

    def subtract_dicts(self, d1, d2):
        """Given nn.state_dicts on the same device, subtract them.

        NOTE: replaces the first dicts info wih the result (in-place)."""
        for key in d1:
            with torch.no_grad():
                d1[key] -= d2[key]
        return d1

    def add_dicts_w_epsilon(self, d1, d2):
        """Again, in place, performs SGD update."""
        for key in d1:
            with torch.no_grad():
                d1[key] += (self.epsilon * d2[key]).to(d1[key].dtype)
        return d1

    def save_checkpoint(self):
        """Copied over from the Runner class.

        NOTE: this doesn't save an optim.
        """
        checkpoint = {
            'nerf': self.nerf_sd,
            'sdf_network_fine': self.sdf_sd,
            'variance_network_fine': self.deviation_sd,
            'color_network_fine': self.color_sd,
            'iter_step': self.iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def writer_write(self, stats):
        for k, v in stats.items():
            self.writer.add_scalar(k, v, self.iter_step)


def read_conf(conf_path, case):
    f = open(conf_path)
    conf_text = f.read()
    conf_text = conf_text.replace('CASE_NAME', case)
    f.close()

    conf = ConfigFactory.parse_string(conf_text)
    return conf


class AverageDict:
    """Just to be compatible with SummaryWriter."""
    def __init__(self):
        self.stats = {}

    def add_scalar(self, name, value, _):
        if name not in self.stats:
            self.stats[name] = []
        self.stats[name].append(float(value))

    def get_summary(self, from_idx=0):
        ret_dict = {}
        for name in self.stats:
            ret_dict[name] = np.array(self.stats[name][from_idx:]).mean()
        return ret_dict


class Runner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', is_continue=False,
                 initial_weights: dict = None, conf=None, no_save=False
                 ):
        self.device = torch.device('cuda')
        if not conf:
            # Configuration
            self.conf_path = conf_path
            f = open(self.conf_path)
            conf_text = f.read()
            conf_text = conf_text.replace('CASE_NAME', case)
            f.close()
            self.conf = ConfigFactory.parse_string(conf_text)
        else:
            self.conf = conf

        self.no_save = no_save
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        # os.makedirs(self.base_exp_dir, exist_ok=True)
        self.dataset = Dataset(self.conf['dataset'])
        self.iter_step = 0

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)

        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None

        # Networks
        params_to_train = []
        self.nerf_outside = NeRF(**self.conf['model.nerf']).to(self.device)
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)
        if initial_weights is not None:
            self.nerf_outside.load_state_dict(initial_weights['nerf'])
            self.sdf_network.load_state_dict(initial_weights['sdf'])
            self.deviation_network.load_state_dict(initial_weights['deviation'])
            self.color_network.load_state_dict(initial_weights['color'])

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

        if not self.no_save:
            # Load checkpoint
            latest_model_name = None
            if is_continue:
                model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
                model_list = []
                for model_name in model_list_raw:
                    if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                        model_list.append(model_name)
                model_list.sort()
                latest_model_name = model_list[-1]

            if latest_model_name is not None:
                logging.info('Find checkpoint: {}'.format(latest_model_name))
                self.load_checkpoint(latest_model_name)

    def get_state_dicts(self, device=None):
        if device is None:
            device = self.device
        return {
            'nerf': self.nerf_outside.to(device).state_dict(),
            'sdf': self.sdf_network.to(device).state_dict(),
            'deviation': self.deviation_network.to(device).state_dict(),
            'color': self.color_network.to(device).state_dict()
        }

    def train(self):
        if self.no_save:
            self.writer = AverageDict()
        else:
            self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))

        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        image_perm = self.get_image_perm()

        for iter_i in range(res_step):
            data = self.dataset.gen_random_rays_at(image_perm[self.iter_step % len(image_perm)], self.batch_size)

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
            render_out = self.renderer.render(rays_o, rays_d, near, far,
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
            psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb)**2 * mask).sum() / (mask_sum * 3.0)).sqrt())

            eikonal_loss = gradient_error

            mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)

            loss = color_fine_loss +\
                   eikonal_loss * self.igr_weight +\
                   mask_loss * self.mask_weight

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_step += 1

            self.writer.add_scalar('Loss/loss', loss, self.iter_step)
            self.writer.add_scalar('Loss/color_loss', color_fine_loss, self.iter_step)
            self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
            self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)
            self.writer.add_scalar('Statistics/cdf', (cdf_fine[:, :1] * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/weight_max', (weight_max * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)

            if not self.no_save:
                if self.iter_step % self.report_freq == 0:
                    print(self.base_exp_dir)
                    print('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, loss, self.optimizer.param_groups[0]['lr']))

                if self.iter_step % self.save_freq == 0:
                    self.save_checkpoint()

                if self.iter_step % self.val_freq == 0:
                    self.validate_image()

                if self.iter_step % self.val_mesh_freq == 0:
                    self.validate_mesh()

            self.update_learning_rate()

            if self.iter_step % len(image_perm) == 0:
                image_perm = self.get_image_perm()

        return self.writer

    def get_image_perm(self):
        return torch.randperm(self.dataset.n_images)

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

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        self.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
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

    def validate_image(self, idx=-1, resolution_level=-1):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
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
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            def feasible(key): return (key in render_out) and (render_out[key] is not None)

            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            if feasible('gradients') and feasible('weights'):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)
            del render_out

        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy())
            normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None])
                          .reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)

        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)

        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'validations_fine',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           np.concatenate([img_fine[..., i],
                                           self.dataset.image_at(idx, resolution_level=resolution_level)]))
            if len(out_normal_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'normals',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           normal_img[..., i])

    def render_novel_image(self, idx_0, idx_1, ratio, resolution_level):
        """
        Interpolate view between two cameras.
        """
        rays_o, rays_d = self.dataset.gen_rays_between(idx_0, idx_1, ratio, resolution_level=resolution_level)
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
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())

            del render_out

        img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255).astype(np.uint8)
        return img_fine

    def validate_mesh(self, world_space=False, resolution=64, threshold=0.0):
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)

        vertices, triangles =\
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))

        logging.info('End')

    def interpolate_view(self, img_idx_0, img_idx_1):
        images = []
        n_frames = 60
        for i in range(n_frames):
            print(i)
            images.append(self.render_novel_image(img_idx_0,
                                                  img_idx_1,
                                                  np.sin(((i / n_frames) - 0.5) * np.pi) * 0.5 + 0.5,
                          resolution_level=4))
        for i in range(n_frames):
            images.append(images[n_frames - i - 1])

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        video_dir = os.path.join(self.base_exp_dir, 'render')
        os.makedirs(video_dir, exist_ok=True)
        h, w, _ = images[0].shape
        writer = cv.VideoWriter(os.path.join(video_dir,
                                             '{:0>8d}_{}_{}.mp4'.format(self.iter_step, img_idx_0, img_idx_1)),
                                fourcc, 30, (w, h))

        for image in images:
            writer.write(image)

        writer.release()


def device_runner(receive_queue, return_queue, conf, device_num):
    """
    Method to call to run a process.

    receive_queue:
        Gets info from the main process used to run an experiment.
    return_queue:
        Returns the weights / stats / anything else to main process.
    conf:
        Used for all experiments.
    """
    print(f"{os.getpid()}: Started meta-learning handle, GPU {device_num}")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(f"cuda:{device_num}")

    while True:
        value = receive_queue.get()
        # Stop flag
        if value is None:
            break
        case, initial_weights = value

        # Train
        print(f"{os.getpid()}: Creating runner with case {case}")
        runner = Runner(
            None, "train", case, False, initial_weights=initial_weights, conf=conf, no_save=True
        )
        print(f"{os.getpid()}: Begin training")
        stats = runner.train()
        print(f"{os.getpid()}: End training, sending weights")

        # Average losses over the last 10 iterations
        stats = stats.get_summary(from_idx=-10)
        # Return on CPU, to prevent OOM (averaging seems very fast on CPU anyway)
        return_queue.put((os.getpid(), runner.get_state_dicts(device="cpu"), stats))


def train_meta_iter(mweights: MetaWeights, send_qs, ret_q, cases):
    """Train for a single iteration."""
    # Send the dataset name and initial weights to all processes
    w = mweights.get_state_dicts()
    for sq, case in zip(send_qs, cases):
        sq.put((case, w))

    # Get the weights and summary from all processes
    sd_infos = []
    stats = AverageDict()
    for _ in range(len(send_qs)):
        # Will deadlock if a sub-process crashes
        pid, info, stat = ret_q.get(block=True)
        print(f"Main: Received weights from {pid}")
        sd_infos.append(info)
        for k, v in stat.items():
            stats.add_scalar(k, v, None)
    stats = stats.get_summary()

    mweights.writer_write(stats)
    mweights.update(sd_infos)
    print("Main: Updated weights")
    return stats


def main():
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='')

    args = parser.parse_args()
    meta_conf = read_conf(args.conf, args.case)
    mweights = MetaWeights(meta_conf)
    mp.set_start_method("spawn")
    print("Sharing: ", torch.multiprocessing.get_sharing_strategy())
    print("Num devices: ", torch.cuda.device_count())

    # Set up processes
    ret_q = mp.Queue()
    send_qs = []
    processes = []
    for idx, d_num in enumerate(range(torch.cuda.device_count())):
        send_qs.append(mp.SimpleQueue())
        processes.append(mp.Process(
            target=device_runner,
            args=(send_qs[-1], ret_q, meta_conf, d_num)
        ))
        print(f"Main: Starting process {idx}")
        processes[-1].start()

    for iter_idx in range(meta_conf["meta.num_outer_iter"]):
        # TODO: Decide how to distribute cases, for now send same to all sub-processes
        cases = ["thin_catbus"] * len(send_qs)

        stats = train_meta_iter(mweights, send_qs, ret_q, cases)
        print(f"{iter_idx}: {stats['Loss/loss']}")

        # TODO: set save freq
        # mweights.save_checkpoint()

    # Shutdown flag
    for sq in send_qs:
        sq.put(None)
    for p in processes:
        p.join()


if __name__ == '__main__':
    print('Hello Wooden')
    main()



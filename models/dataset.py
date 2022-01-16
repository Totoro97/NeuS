import torch
import torch.utils.data
import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
from tqdm import tqdm

import logging
import pathlib
import collections
import random
import os
from glob import glob
import pickle

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


def load_camera_matrices(path):
    camera_dict = np.load(path)

    key_example = 'world_mat_'
    n_images = max(int(x[len(key_example):]) for x in camera_dict if x.startswith(key_example)) + 1

    # world_mat is a projection matrix from world to image
    world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
    # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
    scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]

    intrinsics_all = []
    pose_all = []

    for scale_mat, world_mat in zip(scale_mats_np, world_mats_np):
        P = world_mat @ scale_mat
        P = P[:3, :4]
        intrinsics, pose = load_K_Rt_from_P(None, P)
        intrinsics_all.append(torch.from_numpy(intrinsics).float())
        pose_all.append(torch.from_numpy(pose).float())

    intrinsics_all = torch.stack(intrinsics_all)   # [n_images, 4, 4]
    pose_all = torch.stack(pose_all)  # [n_images, 4, 4]
    focal = intrinsics_all[0][0, 0]

    return pose_all, intrinsics_all, scale_mats_np, focal


class Dataset(torch.utils.data.Dataset):
    def __init__(self, conf, kind='train'):
        super(Dataset, self).__init__()
        logging.info('Load data: Begin')
        self.device = torch.device('cpu')
        self.conf = conf

        # Retrieve config values
        self.data_dirs = conf.get_list('data_dirs')
        self.num_scenes = len(self.data_dirs)
        self.batch_size = conf.get_int('batch_size')

        # Format for `images_to_pick[_val]` in config:
        # [[0, ["00747", "00889"]], [2, ["00053"]], ...]
        images_to_pick_all = conf.get_list(
            {'train': 'images_to_pick', 'val': 'images_to_pick_val'}[kind], default=None)

        # Transpose the above list into
        # [["00747", "00889"], [], ["00053"], ...]
        if images_to_pick_all is None:
            images_to_pick_per_scene = ['default' for _ in range(self.num_scenes)]
        else:
            images_to_pick_per_scene = [[] for _ in range(self.num_scenes)]
            for scene_idx, image_names in images_to_pick_all:
                images_to_pick_per_scene[scene_idx] += image_names

        render_cameras_name = conf.get_string('render_cameras_name')

        def load_one_scene(
            root_dir: pathlib.Path, images_to_pick: list[str] = 'default', kind: str = 'train'):
            """
            images_to_pick
                list of str
                Names of image files (without extension) to keep.
                If 'default', behaviour is controlled by `kind`.
            kind
                str
                Defines behavior when `images_to_pick` is empty.
                If 'train', will load all images in the folder.
                If 'val', will load the first and the last one (in the sorted list of filenames).
            """
            if images_to_pick == []:
                return tuple([] for _ in range(7))

            root_dir = pathlib.Path(root_dir)

            # Load images
            images_list = sorted(x for x in (root_dir / "image").iterdir() if x.suffix == '.png')

            if images_to_pick == 'default':
                images_to_pick = [x.with_suffix('').name for x in images_list]
                if kind == 'val':
                    if len(images_list) >= 2:
                        images_to_pick = [images_to_pick[0], images_to_pick[-1]]
                else:
                    assert kind == 'train', f"Wrong 'kind': '{kind}'"

            def get_image_idx(image_file_name):
                try:
                    return [i for i, x in enumerate(images_list) if x.with_suffix('').name == image_file_name][0]
                except IndexError as exc:
                    raise RuntimeError(f"Asked to pick image '{image_file_name}', couldn't find it") from exc

            image_idxs_to_pick = list(map(get_image_idx, images_to_pick))

            images_list = [images_list[i] for i in image_idxs_to_pick]

            logging.info(f"{kind}, image_idxs_to_pick = {image_idxs_to_pick}")

            n_images = len(images_list)
            # [n_images, H, W, 3], uint8
            images = torch.stack(
                [torch.from_numpy(cv.imread(str(im_name))) for im_name in images_list])

            masks_list = sorted(x for x in (root_dir / "mask").iterdir() if x.suffix == '.png')
            masks_list = [masks_list[i] for i in image_idxs_to_pick]

            # [n_images, H, W, 1], uint8
            masks = torch.stack(
                [torch.from_numpy(cv.imread(str(im_name))[..., 0]) for im_name in masks_list])

            # Load camera parameters
            pose_all, intrinsics_all, scale_mats_np, focal = \
                load_camera_matrices(os.path.join(root_dir, render_cameras_name))
            pose_all = pose_all[image_idxs_to_pick]
            intrinsics_all = intrinsics_all[image_idxs_to_pick]
            scale_mats_np = [scale_mats_np[i] for i in image_idxs_to_pick]

            assert len(pose_all) == n_images
            assert len(intrinsics_all) == n_images
            assert len(scale_mats_np) == n_images

            # Load object bboxes
            with open(root_dir / "tabular_data.pkl", 'rb') as f:
                obj_bboxes = pickle.load(f)['crop_rectangles']
                obj_bboxes = [obj_bboxes[i][:4] for i in image_idxs_to_pick]

            return images, masks, pose_all, intrinsics_all, scale_mats_np, focal, obj_bboxes

        all_data = [load_one_scene(data_dir, images_to_pick, kind=kind) \
            for data_dir, images_to_pick in zip(tqdm(self.data_dirs), images_to_pick_per_scene)]
        # Transpose
        self.images, self.masks, self.pose_all, self.intrinsics_all, \
            self.scale_mats_np, self.focal, self.object_bboxes = zip(*all_data)

        self.pose_all = [x.to(self.device) if x != [] else [] for x in self.pose_all]
        self.intrinsics_all = [x.to(self.device) if x != [] else [] for x in self.intrinsics_all]
        self.intrinsics_all_inv = [torch.inverse(x) if x != [] else [] for x in self.intrinsics_all]

        self.H, self.W = next(filter(lambda x: x != [], self.images)).shape[1:3]
        self.image_pixels = self.H * self.W
        if not all(
            images.shape[1:3] == (self.H, self.W) for images in self.images if images != []):
            raise NotImplementedError("Images of different sizes not supported yet")

        # Region of interest to **extract mesh**
        self.object_bbox_min = np.float32([-1.01, -1.01, -1.01])
        self.object_bbox_max = np.float32([ 1.01,  1.01,  1.01])

        logging.info('Load data: End')

    def __len__(self):
        return self.num_scenes

    def __getitem__(self, i):
        return i, self.gen_random_rays_at(self.batch_size, i)

    def get_image_and_mask(self, scene_idx, image_idx, resolution_level=1):
        # Using nearest neighbors because it's not Mip-NeRF yet
        def resize_and_convert(image, scale):
            image = cv.resize(
                image, dsize=None, fx=scale, fy=scale, interpolation=cv.INTER_NEAREST)
            return torch.from_numpy(image.astype(np.float32) / 255.0)

        scale = 1.0 / resolution_level
        image = resize_and_convert(self.images[scene_idx][image_idx].numpy(), scale)
        mask = resize_and_convert(self.masks[scene_idx][image_idx].numpy(), scale)[..., None]

        return image, mask

    @staticmethod
    def gen_rays(pixels, pose, intrinsics_inv, H, W):
        """
        Compute ray parameters (origin, direction) for `pixels` in a given camera.

        pixels
            torch.Tensor, float32, shape = ([S], 2)
            (x, y) coordinates of pixels
        pose
            torch.Tensor, float32, shape = (3, 4)
            Camera extrinsic parameters (R 3x3 | t 3x1)
        intrinsics_inv
            torch.Tensor, float32, shape = (3, 3)
            Inverse camera projection matrix
        H, W
            int

        return:
        rays_o, rays_v
            torch.Tensor, float32, shape = ([S], 3)
            Rays origin and direction
        """
        assert pixels.shape[-1] == 2
        pixels = torch.cat([pixels, torch.ones_like(pixels[..., :1])], dim=-1) # ([S], 3)

        # Prepare for batched matmul
        intrinsics_inv = intrinsics_inv.reshape(
            (1,) * (pixels.ndim - 1) + intrinsics_inv.shape) # (1, 1, ..., 1, 3, 3)
        pose = pose.reshape(
            (1,) * (pixels.ndim - 1) + pose.shape) # (1, 1, ..., 1, 3, 4)
        pixels = pixels[..., None] # (..., 3, 1)

        pixels = (intrinsics_inv @ pixels).squeeze()  # (..., 3)
        rays_v = pixels / torch.linalg.norm(pixels, ord=2, dim=-1, keepdim=True)  # ([S], 3)
        rays_v = (pose[..., :3] @ rays_v[..., None]).squeeze()  # ([S], 3)
        rays_o = pose[..., 3].expand(rays_v.shape)  # ([S], 1)

        return rays_o, rays_v

    def gen_rays_at(self, scene_idx, image_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels = torch.stack(torch.meshgrid(tx, ty), dim=-1)

        rays_o, rays_v = Dataset.gen_rays(
            pixels, self.pose_all[scene_idx][image_idx, :3, :4],
            self.intrinsics_all_inv[scene_idx][image_idx, :3, :3], self.H, self.W)

        rgb, mask = self.get_image_and_mask(scene_idx, image_idx, resolution_level)

        rays_o = rays_o.transpose(0, 1)
        rays_v = rays_v.transpose(0, 1)
        near, far = self.near_far_from_sphere(rays_o, rays_v)

        return rays_o, rays_v, rgb.to(self.device), mask.to(self.device), near, far

    def gen_random_rays_at(self, batch_size, scene_idx, image_idx=None):
        """
        Generate random rays at world space from one camera.

        image_idx:
            None or int
            If None, sample from 5 random images in scene `scene_idx`.
        """
        remaining_rays_to_sample = batch_size

        # Determine which images to use
        if image_idx is None:
            NUM_IMAGES_TO_USE = 5
            num_images_in_scene = len(self.images[scene_idx])
            num_images_to_use = min(NUM_IMAGES_TO_USE, num_images_in_scene)

            images_idxs_to_use = list(range(num_images_in_scene))
            random.shuffle(images_idxs_to_use)
            images_idxs_to_use = images_idxs_to_use[:num_images_to_use]
            rays_per_image = (batch_size + num_images_to_use - 1) // num_images_to_use
        else:
            images_idxs_to_use = [image_idx]
            rays_per_image = batch_size

        data_to_concat = collections.defaultdict(list)

        for i, current_image_idx in enumerate(images_idxs_to_use):
            rgb, mask = self.get_image_and_mask(scene_idx, current_image_idx)
            l, t, r, b = self.object_bboxes[scene_idx][current_image_idx]

            current_rays_to_sample = min(remaining_rays_to_sample, rays_per_image)
            remaining_rays_to_sample -= current_rays_to_sample

            current_pixels_x = torch.randint(
                low=max(0, l-1), high=min(self.W, r+1), size=[current_rays_to_sample])
            current_pixels_y = torch.randint(
                low=max(0, t-1), high=min(self.H, b+1), size=[current_rays_to_sample])

            rgb = rgb[(current_pixels_y, current_pixels_x)]    # batch_size, 3
            mask = mask[(current_pixels_y, current_pixels_x)]  # batch_size, 1

            current_pixels = torch.stack([current_pixels_x, current_pixels_y], dim=-1).float()
            rays_o, rays_v = Dataset.gen_rays(
                current_pixels, self.pose_all[scene_idx][current_image_idx, :3, :4],
                self.intrinsics_all_inv[scene_idx][current_image_idx, :3, :3], self.H, self.W)

            data_to_concat['rays_o'].append(rays_o)
            data_to_concat['rays_v'].append(rays_v)
            data_to_concat['rgb'].append(rgb)
            data_to_concat['mask'].append(mask)

        for k, v in data_to_concat.items():
            data_to_concat[k] = torch.cat(v)

        near, far = self.near_far_from_sphere(data_to_concat['rays_o'], data_to_concat['rays_v'])

        return tuple(data_to_concat[k] for k in ('rays_o', 'rays_v', 'rgb', 'mask')) + (near, far)

    def gen_rays_between(self, scene_idx, image_idx_0, image_idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[scene_idx][0, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        trans = self.pose_all[scene_idx][image_idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[scene_idx][image_idx_1, :3, 3] * ratio
        pose_0 = self.pose_all[scene_idx][image_idx_0].cpu().numpy()
        pose_1 = self.pose_all[scene_idx][image_idx_1].cpu().numpy()
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
        rot = torch.from_numpy(pose[:3, :3]).to(self.device)
        trans = torch.from_numpy(pose[:3, 3]).to(self.device)
        rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3

        rays_o = rays_o.transpose(0, 1)
        rays_v = rays_v.transpose(0, 1)

        near, far = self.near_far_from_sphere(rays_o, rays_v)

        return rays_o, rays_v, near, far, pose

    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def get_dataloader(self):
        class InfiniteRandomSampler(torch.utils.data.Sampler):
            def __init__(self, dataset_length):
                self.dataset_length = dataset_length

            def __len__(self):
                return 10 ** 20

            def __iter__(self):
                def indices_generator(dataset_length):
                    indices = list(range(dataset_length))

                    while True:
                        random.shuffle(indices)
                        yield from indices

                return indices_generator(self.dataset_length)

        return torch.utils.data.DataLoader(
            self, batch_size=1, num_workers=1,
            sampler=InfiniteRandomSampler(len(self)),
            collate_fn=lambda x: x[0], pin_memory=True)


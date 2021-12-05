import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
from glob import glob
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
from tqdm import tqdm

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


class Dataset:
    def __init__(self, conf):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf

        # Retrieve config values
        self.data_dirs = conf.get_list('data_dirs')
        self.num_scenes = len(self.data_dirs)

        render_cameras_name = conf.get_string('render_cameras_name')

        def load_one_scene(root_dir):
            # Load images
            images_lis = sorted(glob(os.path.join(root_dir, 'image/*.png')))
            n_images = len(images_lis)
            # [n_images, H, W, 3], uint8
            images = np.stack([cv.imread(im_name) for im_name in images_lis])
            masks_lis = sorted(glob(os.path.join(root_dir, 'mask/*.png')))
            # [n_images, H, W, 1], uint8
            masks = np.stack([cv.imread(im_name)[..., 0] for im_name in masks_lis])

            # Load camera parameters
            pose_all, intrinsics_all, scale_mats_np, focal = \
                load_camera_matrices(os.path.join(root_dir, render_cameras_name))

            assert len(pose_all) == n_images
            assert len(intrinsics_all) == n_images
            assert len(scale_mats_np) == n_images

            return images, masks, pose_all, intrinsics_all, scale_mats_np, focal

        all_data = [load_one_scene(data_dir) for data_dir in tqdm(self.data_dirs)]
        # Transpose
        self.images, self.masks, self.pose_all, self.intrinsics_all, _, self.focal = \
            list(zip(*all_data))

        self.pose_all = [x.to(self.device) for x in self.pose_all]
        self.intrinsics_all = [x.to(self.device) for x in self.intrinsics_all]
        self.intrinsics_all_inv = [torch.inverse(x) for x in self.intrinsics_all]

        self.H, self.W = self.images[0].shape[1:3]
        self.image_pixels = self.H * self.W
        if not all(images.shape[1:3] == (self.H, self.W) for images in self.images):
            raise NotImplementedError("Images of different sizes not supported yet")

        # Region of interest to **extract mesh**
        self.object_bbox_min = np.float32([-1.01, -1.01, -1.01])
        self.object_bbox_max = np.float32([ 1.01,  1.01,  1.01])

        print('Load data: End')

    def get_image_and_mask(self, scene_idx, image_idx, resolution_level=1):
        # Using nearest neighbors because it's not Mip-NeRF yet
        def resize_and_convert(image, scale):
            image = cv.resize(
                image, dsize=None, fx=scale, fy=scale, interpolation=cv.INTER_NEAREST)
            return torch.from_numpy(image.astype(np.float32) / 255.0)

        scale = 1.0 / resolution_level
        image = resize_and_convert(self.images[scene_idx][image_idx], scale)
        mask = resize_and_convert(self.masks[scene_idx][image_idx], scale)[..., None]

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

        return rays_o.transpose(0, 1), rays_v.transpose(0, 1), rgb.cuda(), mask.cuda()

    def gen_random_rays_at(self, scene_idx, image_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])

        rgb, mask = self.get_image_and_mask(scene_idx, image_idx)
        rgb = rgb.cuda()[(pixels_y, pixels_x)]    # batch_size, 3
        mask = mask.cuda()[(pixels_y, pixels_x)]  # batch_size, 1

        pixels = torch.stack([pixels_x, pixels_y], dim=-1).float()
        rays_o, rays_v = Dataset.gen_rays(
            pixels, self.pose_all[scene_idx][image_idx, :3, :4],
            self.intrinsics_all_inv[scene_idx][image_idx, :3, :3], self.H, self.W)

        return torch.cat([rays_o, rays_v, rgb, mask], dim=-1)  # batch_size, 10

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
        trans = self.pose_all[scene_idx][image_idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[scene_idx][idx_1, :3, 3] * ratio
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
        rot = torch.from_numpy(pose[:3, :3]).cuda()
        trans = torch.from_numpy(pose[:3, 3]).cuda()
        rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far


import numpy as np
import os
import sys
import cv2 as cv
from glob import glob
from scipy.spatial.transform import Rotation as Rot
from shutil import copytree
import trimesh


def convert_cameras(work_dir, crop_pixels):
    # load cameras
    poses = np.load(os.path.join(work_dir, './tmp/poses.npy'))
    intrinsic_raw = np.load(os.path.join(work_dir, './tmp/intrinsic.npy'))
    n_images = len(poses)

    intrinsic = np.diag([1.0, 1.0, 1.0, 1.0]).astype(np.float32)
    intrinsic[:3, :3] = intrinsic_raw
    intrinsic[0, 2] = intrinsic[0, 2] - crop_pixels
    intrinsic[1, 2] = intrinsic[1, 2] - crop_pixels

    cam_dict = {}

    for i in range(n_images):
        pe = poses[i]
        rot = np.zeros([3, 3])
        cv.Rodrigues(pe[0], rot)
        trans = pe[1]

        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose[:3, :3] = rot
        pose[:3, 3] = trans # w2c

        world_mat = intrinsic @ pose
        world_mat = world_mat.astype(np.float32)
        cam_dict['camera_mat_{}'.format(i)] = intrinsic
        cam_dict['camera_mat_inv_{}'.format(i)] = np.linalg.inv(intrinsic)
        cam_dict['world_mat_{}'.format(i)] = world_mat
        cam_dict['world_mat_inv_{}'.format(i)] = np.linalg.inv(world_mat)

    scale_mat = np.diag([1.0, 1.0, 1.0, 1.0]).astype(np.float32)

    for i in range(n_images):
        cam_dict['scale_mat_{}'.format(i)] = scale_mat
        cam_dict['scale_mat_inv_{}'.format(i)] = np.linalg.inv(scale_mat)

    out_dir = os.path.join(work_dir, 'preprocessed')
    os.makedirs(out_dir, exist_ok=True)

    np.savez(os.path.join(out_dir, 'cameras_sphere.npz'), **cam_dict)

    # provide images
    image_list = glob(os.path.join(work_dir, './tmp/image_undistort/*.png'))
    image_list.sort()

    os.makedirs(os.path.join(out_dir, 'image'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'mask'), exist_ok=True)

    for i, image_path in enumerate(image_list):
        img = cv.imread(image_path)
        img = img[crop_pixels: -crop_pixels, crop_pixels: -crop_pixels]    # remove black area of undistorted images
        cv.imwrite(os.path.join(out_dir, 'image', '{:0>3d}.png'.format(i)), img)
        cv.imwrite(os.path.join(out_dir, 'mask', '{:0>3d}.png'.format(i)), np.ones_like(img) * 255)


if __name__ == '__main__':
    convert_cameras(sys.argv[1], int(sys.argv[2]))

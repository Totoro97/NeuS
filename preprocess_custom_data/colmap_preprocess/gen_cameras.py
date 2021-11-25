import numpy as np
import trimesh
import cv2 as cv
import sys
import os
from glob import glob


if __name__ == '__main__':
    work_dir = sys.argv[1]
    poses_hwf = np.load(os.path.join(work_dir, 'poses.npy')) # n_images, 3, 5
    poses_raw = poses_hwf[:, :, :4]
    hwf = poses_hwf[:, :, 4]
    pose = np.diag([1.0, 1.0, 1.0, 1.0])
    pose[:3, :4] = poses_raw[0]
    pts = []
    pts.append((pose @ np.array([0, 0, 0, 1])[:, None]).squeeze()[:3])
    pts.append((pose @ np.array([1, 0, 0, 1])[:, None]).squeeze()[:3])
    pts.append((pose @ np.array([0, 1, 0, 1])[:, None]).squeeze()[:3])
    pts.append((pose @ np.array([0, 0, 1, 1])[:, None]).squeeze()[:3])
    pts = np.stack(pts, axis=0)
    pcd = trimesh.PointCloud(pts)
    pcd.export(os.path.join(work_dir, 'pose.ply'))
    #

    cam_dict = dict()
    n_images = len(poses_raw)

    # Convert space
    convert_mat = np.zeros([4, 4], dtype=np.float32)
    convert_mat[0, 1] = 1.0
    convert_mat[1, 0] = 1.0
    convert_mat[2, 2] =-1.0
    convert_mat[3, 3] = 1.0

    for i in range(n_images):
        pose = np.diag([1.0, 1.0, 1.0, 1.0]).astype(np.float32)
        pose[:3, :4] = poses_raw[i]
        pose = pose @ convert_mat
        h, w, f = hwf[i, 0], hwf[i, 1], hwf[i, 2]
        intrinsic = np.diag([f, f, 1.0, 1.0]).astype(np.float32)
        intrinsic[0, 2] = (w - 1) * 0.5
        intrinsic[1, 2] = (h - 1) * 0.5
        w2c = np.linalg.inv(pose)
        world_mat = intrinsic @ w2c
        world_mat = world_mat.astype(np.float32)
        cam_dict['camera_mat_{}'.format(i)] = intrinsic
        cam_dict['camera_mat_inv_{}'.format(i)] = np.linalg.inv(intrinsic)
        cam_dict['world_mat_{}'.format(i)] = world_mat
        cam_dict['world_mat_inv_{}'.format(i)] = np.linalg.inv(world_mat)


    pcd = trimesh.load(os.path.join(work_dir, 'sparse_points_interest.ply'))
    vertices = pcd.vertices
    bbox_max = np.max(vertices, axis=0)
    bbox_min = np.min(vertices, axis=0)
    center = (bbox_max + bbox_min) * 0.5
    radius = np.linalg.norm(vertices - center, ord=2, axis=-1).max()
    scale_mat = np.diag([radius, radius, radius, 1.0]).astype(np.float32)
    scale_mat[:3, 3] = center

    for i in range(n_images):
        cam_dict['scale_mat_{}'.format(i)] = scale_mat
        cam_dict['scale_mat_inv_{}'.format(i)] = np.linalg.inv(scale_mat)

    out_dir = os.path.join(work_dir, 'preprocessed')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'image'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'mask'), exist_ok=True)

    image_list = glob(os.path.join(work_dir, 'images/*.png'))
    image_list.sort()

    for i, image_path in enumerate(image_list):
        img = cv.imread(image_path)
        cv.imwrite(os.path.join(out_dir, 'image', '{:0>3d}.png'.format(i)), img)
        cv.imwrite(os.path.join(out_dir, 'mask', '{:0>3d}.png'.format(i)), np.ones_like(img) * 255)

    np.savez(os.path.join(out_dir, 'cameras_sphere.npz'), **cam_dict)
    print('Process done!')

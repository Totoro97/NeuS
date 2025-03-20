import os
import math
import numpy as np
import scipy as sp
import scipy.sparse
import torch
import torch.nn.functional as F

from torch import Tensor

import os
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader, random_split
from glob import glob

# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
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

def preprocess_dataset(scan, device):
    path_to_data = f"./data/DATASET/scan{scan}"
    
    # Load camera parameters
    camera_dict = np.load(os.path.join(path_to_data, 'cameras.npz'))
    
    # Load images and masks
    images_lis = sorted(glob(os.path.join(path_to_data, 'image/*.png')))
    masks_lis = sorted(glob(os.path.join(path_to_data, 'mask/*.png')))
    
    # Load images and masks into numpy arrays
    images_np = np.stack([cv2.imread(im_name) for im_name in images_lis]) / 256.0
    masks_np = np.stack([cv2.imread(im_name) for im_name in masks_lis]) / 256.0
    
    # Load world and scale matrices
    world_mats_np = [camera_dict[f'world_mat_{idx}'].astype(np.float32) for idx in range(len(images_lis))]
    scale_mats_np = [camera_dict[f'scale_mat_{idx}'].astype(np.float32) for idx in range(len(images_lis))]
    
    # Compute intrinsics and poses
    intrinsics_all = []
    pose_all = []
    cond_all = []
    for scale_mat, world_mat in zip(scale_mats_np, world_mats_np):
        P = world_mat @ scale_mat
        P = P[:3, :4]
        intrinsics, pose = load_K_Rt_from_P(None, P)  # Assuming this function is defined
        intrinsics_all.append(torch.from_numpy(intrinsics).float().to(device))
        pose_all.append(torch.from_numpy(pose).float().to(device))
        cond_all.append(torch.cat([intrinsics_all[-1], pose_all[-1]]).flatten())

    # Convert images and masks to PyTorch tensors
    images = torch.from_numpy(images_np.astype(np.float32)).to(device)
    masks = torch.from_numpy(masks_np.astype(np.float32)).to(device)
    
    # Stack intrinsics and poses
    intrinsics_all = torch.stack(intrinsics_all)
    pose_all = torch.stack(pose_all)
    cond_all = torch.stack(cond_all)
    
    # Create a dataset object (custom class)
    class ImageDataset(torch.utils.data.Dataset):
        def __init__(self, images, masks, intrinsics_all, pose_all, cond_all):
            self.images = images
            self.masks = masks
            self.intrinsics_all = intrinsics_all
            self.pose_all = pose_all
            self.cond_all = cond_all
            
        def __len__(self):
            return len(self.images)
        
        def __getitem__(self, idx):
            return {
                'image': self.images[idx],
                'mask': self.masks[idx],
                'intrinsics': self.intrinsics_all[idx],
                'pose': self.pose_all[idx],
                'cond': self.cond_all[idx]
            }
    
    # Create the full dataset
    full_dataset = ImageDataset(images, masks, intrinsics_all, pose_all, cond_all)
    
    # Split into train, validation, and test sets (e.g., 70% train, 15% val, 15% test)
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    trainset, valset, testset = random_split(full_dataset, [train_size, val_size, test_size])
    
    return trainset, valset, testset



def handle_nan(x):
    if math.isnan(x):
        return float(-100)
    return x




def masked_instance_norm2D(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-5):
    """
    x: [batch_size (N), num_objects (L), num_objects (L), features(C)]
    mask: [batch_size (N), num_objects (L), num_objects (L), 1]
    """
    mask = mask.view(x.size(0), x.size(1), x.size(2), 1).expand_as(x)
    mean = (torch.sum(x * mask, dim=[1,2]) / torch.sum(mask, dim=[1,2]))   # (N,C)
    var_term = ((x - mean.unsqueeze(1).unsqueeze(1).expand_as(x)) * mask)**2  # (N,L,L,C)
    var = (torch.sum(var_term, dim=[1,2]) / torch.sum(mask, dim=[1,2]))  # (N,C)
    mean = mean.unsqueeze(1).unsqueeze(1).expand_as(x)  # (N, L, L, C)
    var = var.unsqueeze(1).unsqueeze(1).expand_as(x)    # (N, L, L, C)
    instance_norm = (x - mean) / torch.sqrt(var + eps)   # (N, L, L, C)
    instance_norm = instance_norm * mask
    return instance_norm


def masked_layer_norm2D(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-5):
    """
    x: [batch_size (N), num_objects (L), num_objects (L), features(C)]
    mask: [batch_size (N), num_objects (L), num_objects (L), 1]
    """
    mask = mask.view(x.size(0), x.size(1), x.size(2), 1).expand_as(x)
    mean = torch.sum(x * mask, dim=[3,2,1]) / torch.sum(mask, dim=[3,2,1])   # (N)
    var_term = ((x - mean.view(-1,1,1,1).expand_as(x)) * mask)**2  # (N,L,L,C)
    var = (torch.sum(var_term, dim=[3,2,1]) / torch.sum(mask, dim=[3,2,1]))  # (N)
    mean = mean.view(-1,1,1,1).expand_as(x)  # (N, L, L, C)
    var = var.view(-1,1,1,1).expand_as(x)    # (N, L, L, C)
    layer_norm = (x - mean) / torch.sqrt(var + eps)   # (N, L, L, C)
    layer_norm = layer_norm * mask
    return layer_norm


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start






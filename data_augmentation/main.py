import argparse
import os
import random
import scipy as sp
import pickle

import shutil
import csv
import ast

import scipy.sparse as sparse
from tqdm import tqdm
from torch import Tensor
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.utils.data import DataLoader

from denoise_model import DenoiseNN, p_losses, sample
from utils import linear_beta_schedule, preprocess_dataset

from torch.utils.data import Subset
np.random.seed(13)

# Argument parser
parser = argparse.ArgumentParser(description='ImageGenerator')

# Argument parser for configuring the NeuralGraphGenerator model
parser = argparse.ArgumentParser(description='Configuration for the ImageGenerator model')

# Learning rate for the optimizer
parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate for the optimizer, typically a small float value (default: 0.001)")

# Dropout rate
parser.add_argument('--dropout', type=float, default=0.0, help="Dropout rate (fraction of nodes to drop) to prevent overfitting (default: 0.0)")

# Batch size for training
parser.add_argument('--batch-size', type=int, default=256, help="Batch size for training, controlling the number of samples per gradient update (default: 256)")

# Number of training epochs for the denoising model
parser.add_argument('--epochs-denoise', type=int, default=100, help="Number of training epochs for the denoising model (default: 100)")

# Number of timesteps in the diffusion
parser.add_argument('--timesteps', type=int, default=500, help="Number of timesteps for the diffusion (default: 500)")

# Hidden dimension size for the denoising model
parser.add_argument('--hidden-dim-denoise', type=int, default=512, help="Hidden dimension size for denoising model layers (default: 512)")

# Number of layers in the denoising model
parser.add_argument('--n-layers_denoise', type=int, default=3, help="Number of layers in the denoising model (default: 3)")

# Flag to toggle training of the diffusion-based denoising model
parser.add_argument('--train-denoiser', action='store_true', default=True, help="Flag to enable/disable denoiser training (default: enabled)")

# Scan number from DTU dataset
parser.add_argument('--scan', type=int, default=83, help="Scan number from DTU dataset")

# Dimensionality of conditioning vectors for conditional generation
parser.add_argument('--dim-condition', type=int, default=128, help="Dimensionality of conditioning vectors for conditional generation (default: 128)")

# Early-Stopping
parser.add_argument('--patience', type=int, default=20, help="Number of epoch with no improvement leading to early stop of the training.")

args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# preprocess train data, validation data and test data. Only once for the first time that you run the code. Then the appropriate .pt files will be saved and loaded.
trainset, validset, testset = preprocess_dataset(args.scan, device)

# initialize data loaders
train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(validset, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

# define beta schedule
betas = linear_beta_schedule(timesteps=args.timesteps)

# define alphas
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

denoise_model = DenoiseNN(
    input_channels=trainset[0]['image'].shape[-1], 
    hidden_dim=16, 
    n_layers=2, 
    n_cond=trainset[0]['cond'].shape[-1], 
    d_cond=8
).to(device)

optimizer = torch.optim.Adam(denoise_model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

metrics_denoise = {'train_loss': [], 'val_loss': []}

# Train denoising model
if args.train_denoiser:
    best_val_loss = np.inf
    wait = 0
    for epoch in range(1, args.epochs_denoise+1):
        denoise_model.train()
        train_loss_all = 0
        train_count = 0
        for data in train_loader:
            optimizer.zero_grad()
            x = data['image']
            t = torch.randint(0, args.timesteps, (x.size(0),), device=device).long()
            
            loss = p_losses(denoise_model, x, t, data['cond'], sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, loss_type="huber", mask=data['mask'])
                        
            loss.backward()
            
            train_loss_all += x.size(0) * loss.item()
            train_count += x.size(0)
            optimizer.step()

        denoise_model.eval()
        val_loss_all = 0
        val_count = 0
        for data in val_loader:
            x = data['image']
            t = torch.randint(0, args.timesteps, (x.size(0),), device=device).long()
            
            loss = p_losses(denoise_model, x, t, data['cond'], sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, loss_type="huber", mask=data['mask'])
            
            val_loss_all += x.size(0) * loss.item()
            val_count += x.size(0)

        metrics_denoise['train_loss'].append(train_loss_all/train_count)
        metrics_denoise['val_loss'].append(val_loss_all/val_count)
        
        if epoch % 5 == 0:
            dt_t = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print('{} Epoch: {:04d}, Train Loss: {:.5f}, Val Loss: {:.5f}'.format(dt_t, epoch, train_loss_all/train_count, val_loss_all/val_count))

        scheduler.step()

        if best_val_loss > val_loss_all:
            wait = 0
            best_val_loss = val_loss_all
            torch.save({
                'state_dict': denoise_model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, 'denoise_model.pth.tar')
        else:
            wait += 1
        
        if wait >= args.patience:
            break
else:
    checkpoint = torch.load('denoise_model.pth.tar')
    denoise_model.load_state_dict(checkpoint['state_dict'])

denoise_model.eval()

del train_loader, val_loader

# Save metrics
import json
dt_t = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
with open(f"metrics_{dt_t}.json", "w") as f:
    json.dump({"denoise": metrics_denoise}, f)
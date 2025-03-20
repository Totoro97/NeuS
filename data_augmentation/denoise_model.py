import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


# forward diffusion (using the nice property)
def q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


# Loss function for denoising
def p_losses(denoise_model, x_start, t, cond, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None, loss_type="l1", mask=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=noise)
    predicted_noise = denoise_model(x_noisy, t, cond)

    if mask:
        predicted_noise = predicted_noise * mask
        noise = noise * mask

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss


# Position embeddings
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


import torch
import torch.nn as nn
import torch.nn.functional as F

class DenoiseNN(nn.Module):
    def __init__(self, input_channels, hidden_dim, n_layers, n_cond, d_cond):
        super(DenoiseNN, self).__init__()
        self.n_layers = n_layers
        self.n_cond = n_cond
        self.input_channels = input_channels

        # Conditioning MLP
        self.cond_mlp = nn.Sequential(
            nn.Linear(n_cond, d_cond),
            nn.ReLU(),
            nn.Linear(d_cond, d_cond),
        )

        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Convolutional layers for image processing
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(nn.Conv2d(input_channels + d_cond, hidden_dim, kernel_size=3, padding=1))
        for _ in range(n_layers - 2):
            self.conv_layers.append(nn.Conv2d(hidden_dim + d_cond, hidden_dim, kernel_size=3, padding=1))
        self.conv_layers.append(nn.Conv2d(hidden_dim + d_cond, input_channels, kernel_size=3, padding=1))

        # Batch normalization layers
        self.bn_layers = nn.ModuleList([nn.BatchNorm2d(hidden_dim) for _ in range(n_layers - 1)])

        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.05)

    def forward(self, x, t, cond):
        # x: (batch_size, H, W, 3)
        # cond: (batch_size, n_cond)
        # t: (batch_size,)

        # Reshape x to (batch_size, 3, H, W)
        x = x.permute(0, 3, 1, 2)

        # Process conditioning
        cond = torch.reshape(cond, (-1, self.n_cond))
        cond = torch.nan_to_num(cond, nan=-100.0)
        cond = self.cond_mlp(cond)  # (batch_size, d_cond)

        # Process time embedding
        t = self.time_mlp(t)  # (batch_size, hidden_dim)

        # Reshape cond and t to match spatial dimensions of x
        cond = cond.unsqueeze(-1).unsqueeze(-1)  # (batch_size, d_cond, 1, 1)
        cond = cond.expand(-1, -1, x.size(2), x.size(3))  # (batch_size, d_cond, H, W)

        t = t.unsqueeze(-1).unsqueeze(-1)  # (batch_size, hidden_dim, 1, 1)
        t = t.expand(-1, -1, x.size(2), x.size(3))  # (batch_size, hidden_dim, H, W)

        # Concatenate cond with x
        x = torch.cat([x, cond], dim=1)  # (batch_size, input_channels + d_cond, H, W)

        # Apply convolutional layers
        for i in range(self.n_layers - 1):
            x = self.conv_layers[i](x)
            x = self.relu(x) + t  # Add time embedding
            x = self.bn_layers[i](x)
            x = self.dropout(x)
            if i < self.n_layers - 2:
                x = torch.cat([x, cond], dim=1)  # Concatenate cond for next layer

        # Final layer
        x = self.conv_layers[-1](x)  # (batch_size, input_channels, H, W)

        # Reshape back to (batch_size, H, W, 3)
        x = x.permute(0, 2, 3, 1)

        return x


@torch.no_grad()
def p_sample(model, x, t, cond, t_index, betas):
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

    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t, cond) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise

# Algorithm 2 (including returning all images)
@torch.no_grad()
def p_sample_loop(model, cond, timesteps, betas, shape):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in reversed(range(0, timesteps)):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), cond, i, betas)
        imgs.append(img)
        #imgs.append(img.cpu().numpy())
    return imgs



@torch.no_grad()
def sample(model, cond, latent_dim, timesteps, betas, batch_size):
    return p_sample_loop(model, cond, timesteps, betas, shape=(batch_size, latent_dim))

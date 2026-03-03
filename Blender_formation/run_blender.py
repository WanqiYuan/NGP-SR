import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import datetime
import torch.nn.functional as F
import math



from torchvision import datasets
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.utils import save_image
from torchvision import transforms
from torchvision.utils import make_grid
import torchvision.utils as vutils
from itertools import product
from tqdm import tqdm
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


import json
import os
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import csv
import datetime

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


from torch.cuda.amp import autocast, GradScaler

import torch
from torch.cuda.amp import GradScaler

def _amp_dtype():
    try:
        if torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:
        pass
    return torch.float16

try:
    from torch import autocast as _torch_autocast

    def autocast_cm(enabled: bool, amp_dtype: torch.dtype):
        return _torch_autocast("cuda", dtype=amp_dtype, enabled=enabled)

except Exception:
    from torch.cuda.amp import autocast as _cuda_autocast

    def autocast_cm(enabled: bool, amp_dtype: torch.dtype):
        return _cuda_autocast(enabled=enabled, dtype=amp_dtype)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NeRFDataset(Dataset):
    def __init__(self, root, split='train', transform=None, target_transform=None, loader=None, finest_resolution=None):

        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader if loader is not None else self.default_loader
        self.img_dir = os.path.join(root, 'images')
        self.finest_resolution = finest_resolution
        
        #read json
        json_path = os.path.join(root, f'transforms_{split}.json')
        with open(json_path, 'r') as f:
            data = json.load(f)
            self.camera_angle_x = data.get('camera_angle_x', None)
            self.frames = data.get('frames', [])

    def __getitem__(self, index):

        frame = self.frames[index]
    
        
        img_path = os.path.join(self.root, frame['file_path'] + '.png')  

        img = self.loader(img_path)
        if self.transform is not None:
            lr_img = self.transform(img)  
        else:
            lr_img = transforms.ToTensor()(img)
        
        if self.finest_resolution is not None:
            hr_transform = transforms.Compose([
                transforms.Resize((self.finest_resolution, self.finest_resolution)),
                transforms.ToTensor()
            ])
            hr_img = hr_transform(img)
        else:
            hr_img = lr_img.clone()

        camera_data = {
            'transform_matrix': np.array(frame['transform_matrix'], dtype=np.float32),
            'file_path': frame['file_path']  
        }

        if self.camera_angle_x is not None:
            camera_data['camera_angle_x'] = self.camera_angle_x

        return lr_img, hr_img, camera_data

    def __len__(self):
        return len(self.frames)

    def default_loader(self, path):
        img = Image.open(path)
        if img.mode != 'RGBA':
            img = img.convert('RGBA')

        background = Image.new('RGB', img.size, (255, 255, 255))
        if img.mode == 'RGBA':
            background.paste(img, mask=img.split()[3])
            img = background
        else:
            img = img.convert('RGB')
        return img

def get_nerf_datasets(root, transform=None, target_transform=None, finest_resolution=None):

    train_dataset = NeRFDataset(root, 'train', transform, target_transform, finest_resolution=finest_resolution)
    val_dataset = NeRFDataset(root, 'val', transform, target_transform, finest_resolution=finest_resolution)
    test_dataset = NeRFDataset(root, 'test', transform, target_transform, finest_resolution=finest_resolution)
    
    return train_dataset, val_dataset, test_dataset

class CameraPoseEncoder(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=64, output_dim=1):
        super(CameraPoseEncoder, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pose_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        ).to(self.device)
        for layer in self.pose_encoder:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                nn.init.zeros_(layer.bias)
    def forward(self, camera_matrix):
        batch_size = camera_matrix.shape[0]
        flattened_matrix = camera_matrix.reshape(batch_size, -1)
        encoded = self.pose_encoder(flattened_matrix)
        return encoded
def get_patch_index(height, width, array_of_coords, grid_resolution, patch_height, patch_width):
    array_of_coords = torch.clamp(array_of_coords, 0, 1 - 1e-7)
    array_of_coords = array_of_coords * torch.tensor([width, height], dtype=torch.float32).view(1, 2)
    col_indices = (array_of_coords[:, 0] / patch_width).long()
    row_indices = (array_of_coords[:, 1] / patch_height).long()
    col_indices = torch.clamp(col_indices, 0, grid_resolution - 1)
    row_indices = torch.clamp(row_indices, 0, grid_resolution - 1)
    patch_indices = row_indices * grid_resolution + col_indices
    max_index = grid_resolution**2 - 1
    patch_indices = torch.clamp(patch_indices, 0, max_index)
    return patch_indices
def prepare_array_of_coord(h, w, SR_factor=1, activation='sigmoid'):
    h, w = int(h*SR_factor), int(w*SR_factor)
    if activation == 'tanh':
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, steps=h),
            torch.linspace(-1, 1, steps=w)
        )
    else:
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, steps=h),
            torch.linspace(0, 1, steps=w)
        )
    coords = torch.stack((grid_x, grid_y), dim=-1).view(-1, 2)
    return coords
def prepare_coords(h, w, t, coords_dim=3, SR_factor=1, activation='tanh'):
    h, w = int(h*SR_factor), int(w*SR_factor)
    if activation == 'tanh':
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, steps=h),
            torch.linspace(-1, 1, steps=w)
        )
    else:
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, steps=h),
            torch.linspace(0, 1, steps=w)
        )
    if coords_dim == 3:
        grid_t = torch.full((h, w), t)
        coords = torch.stack((grid_x, grid_y, grid_t), dim=-1)
    elif coords_dim == 2:
        coords = torch.stack((grid_x, grid_y), dim=-1)
    else:
        raise ValueError("Invalid coords_dim. Must be 2 or 3.")
    return coords
def matrix_to_quaternion(matrix):
    trace = matrix.diagonal().sum()
    if trace > 0:
        S = torch.sqrt(trace + 1.0) * 2
        w = 0.25 * S
        x = (matrix[2, 1] - matrix[1, 2]) / S
        y = (matrix[0, 2] - matrix[2, 0]) / S
        z = (matrix[1, 0] - matrix[0, 1]) / S
    else:
        if matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
            S = torch.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
            w = (matrix[2, 1] - matrix[1, 2]) / S
            x = 0.25 * S
            y = (matrix[0, 1] + matrix[1, 0]) / S
            z = (matrix[0, 2] + matrix[2, 0]) / S
        elif matrix[1, 1] > matrix[2, 2]:
            S = torch.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
            w = (matrix[0, 2] - matrix[2, 0]) / S
            x = (matrix[0, 1] + matrix[1, 0]) / S
            y = 0.25 * S
            z = (matrix[1, 2] + matrix[2, 1]) / S
        else:
            S = torch.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
            w = (matrix[1, 0] - matrix[0, 1]) / S
            x = (matrix[0, 2] + matrix[2, 0]) / S
            y = (matrix[1, 2] + matrix[2, 1]) / S
            z = 0.25 * S
    return torch.stack([w, x, y, z])
def prepare_coords_with_view(h, w, camera_data, coords_dim=3, SR_factor=1, activation='tanh', camera_encoder=None):
    h, w = int(h*SR_factor), int(w*SR_factor)
    device = next(camera_encoder.parameters()).device if camera_encoder is not None else 'cpu'
    if activation == 'tanh':
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, steps=h, device=device),
            torch.linspace(-1, 1, steps=w, device=device)
        )
    else:
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, steps=h, device=device),
            torch.linspace(0, 1, steps=w, device=device)
        )
    if isinstance(camera_data['transform_matrix'], torch.Tensor):
        transform_matrix = camera_data['transform_matrix'].to(device)
    else:
        transform_matrix = torch.tensor(camera_data['transform_matrix'], device=device)
    rotation_matrix = transform_matrix[:3, :3]
    translation = transform_matrix[:3, 3]
    quaternion = matrix_to_quaternion(rotation_matrix)
    pose_vector = torch.cat([translation, quaternion])
    r = torch.norm(translation)
    theta = torch.atan2(translation[1], translation[0])
    phi = torch.acos(translation[2] / r)
    r_norm = 2 * (r / (r + 1)) - 1
    theta_norm = theta / torch.pi
    phi_norm = (phi / torch.pi) * 2 - 1
    angle = 2 * torch.acos(torch.abs(quaternion[0]))
    angle_norm = angle / torch.pi
    axis = quaternion[1:]
    axis_norm = axis / (torch.norm(axis) + 1e-8)
    pose_features = torch.stack([
        r_norm,
        theta_norm,
        phi_norm,
        angle_norm * axis_norm[0],
        angle_norm * axis_norm[1],
        angle_norm * axis_norm[2]
    ])
    weights = torch.tensor([0.3, 0.2, 0.2, 0.1, 0.1, 0.1], device=device)
    p = torch.sum(pose_features * weights)
    p = torch.clamp(p, -1, 1)
    grid_x_flat = grid_x.reshape(-1)
    grid_y_flat = grid_y.reshape(-1)
    p_repeated = torch.full_like(grid_x_flat, p)
    if coords_dim == 3:
        coords = torch.stack((grid_x_flat, grid_y_flat, p_repeated), dim=-1)
    elif coords_dim == 2:
        coords = torch.stack((grid_x_flat, grid_y_flat), dim=-1)
    else:
        raise ValueError("Invalid coords_dim. Must be 2 or 3.")
    return coords.reshape(h, w, -1)
def generate_resolutions_list(base_resolution, finest_resolution, n_levels):
    if n_levels == 1:
        return [finest_resolution]
    step = (finest_resolution - base_resolution) / (n_levels - 1)
    resolutions_list = []
    for i in range(n_levels):
        resolution = int(base_resolution + i * step)
        resolutions_list.append(resolution)
    return resolutions_list
class DownsampleDatasetWithCoords(Dataset):
    def __init__(self, dataset, downscale_factor_list, resolutions_list, n_levels, n_pixels,
                 finest_resolution=128, activation='tanh', coords_dim=3,SR_factor=1,
                 SR_INPUT=False, camera_encoder=None):
        self.dataset = dataset
        self.downscale_factor_list = downscale_factor_list
        self.downsample_transforms = [
            transforms.Resize(
                (finest_resolution // factor, finest_resolution // factor)
            )
            for factor in downscale_factor_list
        ]
        self.upsample_transform = transforms.Resize(
            (finest_resolution, finest_resolution)
        )
        self.finest_resolution = finest_resolution
        self.activation = activation
        self.coords_dim = coords_dim
        self.SR_INPUT = SR_INPUT
        self.SR_factor = SR_factor
        self.camera_encoder = camera_encoder
        self.n_pixel_per_batch = n_pixels
        self.n_levels = n_levels
        self.resolutions = resolutions_list
        self.patch_heights = []
        self.patch_widths = []
        for i in range(n_levels):
            resolution = self.resolutions[i]
            patch_height = finest_resolution // resolution
            patch_width = finest_resolution // resolution
            self.patch_heights.append(patch_height)
            self.patch_widths.append(patch_width)
    def __len__(self):
        return len(self.dataset) * len(self.downscale_factor_list)
    def __getitem__(self, idx, IDX_OFFSET=0):
        dataset_idx = idx // len(self.downscale_factor_list)
        scale_idx = idx % len(self.downscale_factor_list)
        image, camera_data = self.dataset[dataset_idx]
        C, H, W = image.shape
        downsample_transform = self.downsample_transforms[scale_idx]
        image_size = (C, H, W)
        save_idx = dataset_idx + IDX_OFFSET
        file_path = camera_data['file_path']
        file_name = file_path.split('/')[-1]
        downsampled_image = downsample_transform(image)
        input_image_lr = self.upsample_transform(downsampled_image)
        upscaled_height = int(H * self.SR_factor)
        upscaled_width = int(W * self.SR_factor)
        upsample_transform = transforms.Resize((upscaled_height, upscaled_width))
        if self.SR_INPUT:
            sr_image = upsample_transform(image)
            coord_sr_scale = self.SR_factor
        else:
            sr_image = image
            coord_sr_scale = 1
        coords_input = prepare_coords_with_view(
            H, W,
            camera_data,
            coords_dim=self.coords_dim,
            activation=self.activation,
            SR_factor=coord_sr_scale,
            camera_encoder=self.camera_encoder
        )
        patch_features = []
        patch_index = []
        camera_matrix = torch.tensor(camera_data['transform_matrix'], dtype=torch.float32)
        for i in range(self.n_levels):
            resolution = self.resolutions[i]
            patches = input_image_lr.unfold(1, self.patch_heights[i], self.patch_heights[i])\
                .unfold(2, self.patch_widths[i], self.patch_widths[i])
            C, num_patches_y, num_patches_x, ph, pw = patches.shape
            patch_indices = get_patch_index(
                self.finest_resolution, self.finest_resolution,
                prepare_array_of_coord(H, W, SR_factor=coord_sr_scale),
                resolution, num_patches_y, num_patches_x
            )
            patches = patches.permute(1, 2, 0, 3, 4).reshape(num_patches_y * num_patches_x, -1)
            patch_with_camera = {
                'patches': patches,
                'camera_matrix': camera_matrix
            }
            patch_features.append(patch_with_camera)
            patch_index.append(patch_indices)
        coords_input_batches = torch.split(coords_input.view(-1, self.coords_dim), self.n_pixel_per_batch)
        HR_img_batches = torch.split(sr_image.reshape(-1, C), self.n_pixel_per_batch)
        patch_index_batches = [torch.split(index_list, self.n_pixel_per_batch) for index_list in patch_index]
        data_batches = []
        for batch_idx in range(len(coords_input_batches)):
            batch_coords = coords_input_batches[batch_idx]
            batch_hrs = HR_img_batches[batch_idx]
            batch_features = [features_batch[batch_idx] for features_batch in patch_index_batches]
            data_batches.append((batch_features, batch_coords, batch_hrs, camera_matrix))
        return data_batches, patch_features, input_image_lr.reshape(-1, C), image_size, file_name
def concatenate_images_to_grid(images, cols=8):
    batch_size = images.shape[0]
    rows = (batch_size + cols - 1) // cols
    grid_image = vutils.make_grid(images, nrow=cols, padding=0, normalize=True, scale_each=True)
    return grid_image[None,:]
def visualize_coordinates(coords, title="Coordinate Visualization", DISPLAY_COL=None):
    B, _, H, W = coords.shape
    coords_normalized = (coords + 1) / 2
    color_values = torch.ones((B, 1, H, W))
    coords_with_color = torch.cat((coords_normalized, color_values), dim=1)
    grid_image = concatenate_images_to_grid(coords_with_color, DISPLAY_COL)
    plt.figure(figsize=(12, 12))
    plt.imshow(grid_image.squeeze(0).numpy().transpose(1, 2, 0))
    plt.axis('off')
    plt.title(title)
    plt.show()
class FullyConnectedLayer(nn.Module):
    def __init__(self,
        in_features,
        out_features,
        bias            = True,
        activation      = 'linear',
        lr_multiplier   = 1,
        bias_init       = 0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier
    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain
        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act(x, b, act=self.activation)
        return x
    def extra_repr(self):
        return f'in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s}'
def bias_act(x, b=None, act='linear'):
    if b is not None:
        x = x + b
    if act == 'relu':
        x = torch.relu(x)
    elif act == 'lrelu':
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.2)
    elif act == 'sigmoid':
        x = torch.sigmoid(x)
    elif act == 'tanh':
        x = torch.tanh(x)
    return x
def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()
def hash(coords, log2_hashmap_size=20):
    primes = torch.tensor([
        1, 11614769, 2654435761, 805459861, 2246822519, 3266489917,
        4294967291, 1500450271, 1459629363, 273326509,  12341647, 13363367,
        15208767, 16430317, 17425307, 20394401, 21785619, 17249767,
        22039621, 23488747, 23879561, 24354221, 25157537, 25928687,
        26685441, 27144023, 27560431, 28074179, 28957989, 29674693,
        30424371, 31207067
    ][:coords.size(-1)], device=coords.device)
    xor_result = torch.zeros_like(coords)[..., 0]
    for i in range(coords.size(-1)):
        xor_result ^= coords[..., i] * primes[i]
    return torch.tensor((1 << log2_hashmap_size) - 1).to(xor_result.device) & xor_result
class BestHashPicker(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super(BestHashPicker, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
    def forward(self, primes):
        return self.network(primes.float())
class CameraAwareInterpolator(nn.Module):
    def __init__(self,
        input_dim,
        camera_pe_size,
        hidden_dim,
        output_dim,
        embed_features  = None,
        num_layers      = 3,
        activation      = 'tanh',
        lr_multiplier   = 1,
        normalize_input = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.camera_pe_size = camera_pe_size
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.normalize_input = normalize_input
        self.activation = activation
        if embed_features is None:
            embed_features = hidden_dim
        self.embed_features = embed_features
        self.camera_embed = FullyConnectedLayer(camera_pe_size, embed_features,
                                               activation='tanh', lr_multiplier=lr_multiplier)
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'lrelu':
            self.act = nn.LeakyReLU(negative_slope=0.2)
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        else:
            self.act = nn.Identity()
        self.norm_layers = nn.ModuleList()
        if num_layers == 1:
            features_list = [input_dim + embed_features, output_dim]
        else:
            features_list = [input_dim + embed_features] + [hidden_dim + embed_features] * (num_layers - 2) + [hidden_dim + embed_features, output_dim]
            for _ in range(num_layers - 1):
                self.norm_layers.append(nn.LayerNorm(hidden_dim))
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_features = features_list[i]
            out_features = hidden_dim if i < num_layers - 1 else output_dim
            layer = FullyConnectedLayer(
                in_features,
                out_features,
                activation='tanh',
                lr_multiplier=lr_multiplier
            )
            self.layers.append(layer)
    def forward(self, x, camera_pe):
        batch_size, num_points, input_dim = x.shape
        x_flat = x.reshape(batch_size * num_points, input_dim)
        if self.normalize_input:
            x_flat = normalize_2nd_moment(x_flat)
        camera_embed = self.camera_embed(camera_pe)
        if self.normalize_input:
            camera_embed = normalize_2nd_moment(camera_embed)
        camera_embed_expanded = camera_embed.unsqueeze(1).expand(-1, num_points, -1)
        camera_embed_flat = camera_embed_expanded.reshape(batch_size * num_points, -1)
        features = torch.cat([x_flat, camera_embed_flat], dim=1)
        for i, layer in enumerate(self.layers):
            if i == self.num_layers - 1:
                features = layer(features)
            else:
                hidden = layer(features)
                hidden = self.act(hidden)
                if i < len(self.norm_layers):
                    hidden = self.norm_layers[i](hidden)
                features = torch.cat([hidden, camera_embed_flat], dim=1)
        if self.activation == 'tanh':
            output = torch.tanh(features)
        else:
            output = torch.sigmoid(features)
        output = output.reshape(batch_size, num_points, self.output_dim)
        return output
    def extra_repr(self):
        return f'input_dim={self.input_dim:d}, camera_pe_size={self.camera_pe_size:d}, embed_features={self.embed_features:d}, hidden_dim={self.hidden_dim:d}, output_dim={self.output_dim:d}, num_layers={self.num_layers:d}, activation={self.activation:s}'
class HashEmbedder(nn.Module):
    def __init__(self, resolutions_list, bounding_box, n_levels=16, n_features_per_level=2,
                 log2_hashmap_size=19, base_resolution=16, finest_resolution=512, mode='NN',
                 num_mf_layers=1):
        super(HashEmbedder, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bounding_box = bounding_box
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = torch.tensor(base_resolution)
        self.finest_resolution = torch.tensor(finest_resolution)
        self.num_mf_layers = num_mf_layers
        self.out_dim = num_mf_layers * self.n_features_per_level
        self.n_dim = bounding_box[0].size(0)
        self.mode = mode
        self.resolutions = resolutions_list
        self.mf_layer_indices = self._get_mf_layer_indices(num_mf_layers)
        self.mf_resolutions = {}
        self.mf_embeddings = nn.ModuleDict()
        for idx, layer_idx in enumerate(self.mf_layer_indices):
            key = f'MF{idx}'
            resolution = int(self.resolutions[layer_idx])
            self.mf_resolutions[key] = resolution
            self.mf_embeddings[key] = nn.Embedding(2**self.log2_hashmap_size, self.n_features_per_level)
            nn.init.uniform_(self.mf_embeddings[key].weight, a=-0.0001, b=0.0001)
        if mode == 'interp':
            self.create_weight = CameraAwareInterpolator(
                input_dim=3 * self.n_dim,
                camera_pe_size=16,
                hidden_dim=64,
                output_dim=2**self.n_dim,
                embed_features=32,
                num_layers=3,
                activation='tanh',
                lr_multiplier=1
            )
        elif mode == 'NN':
            self.create_weight = CameraAwareInterpolator(
                input_dim=2**self.n_dim * n_features_per_level,
                camera_pe_size=16,
                hidden_dim=64,
                output_dim=2**self.n_dim,
                embed_features=32,
                num_layers=3,
                activation='tanh',
                lr_multiplier=1
            )
        self.primes = torch.tensor([
            1, 11614769, 2654435761, 805459861, 2246822519, 3266489917,
            4294967291, 1500450271, 1459629363, 273326509,  12341647, 13363367,
            15208767, 16430317, 17425307, 20394401, 21785619, 17249767,
            22039621, 23488747, 23879561, 24354221, 25157537, 25928687,
            26685441, 27144023, 27560431, 28074179, 28957989, 29674693,
            30424371, 31207067
        ], device=self.device)
        self.hash_picker = BestHashPicker(input_dim=len(self.primes), hidden_dim=64, output_dim=len(self.primes)).to(self.device)
    def _get_mf_layer_indices(self, num_mf_layers):
        print("number of mf")
        print(num_mf_layers)
        if num_mf_layers == 1:
            return [11]
        elif num_mf_layers == 2:
            return [5, 11]
        elif num_mf_layers == 3:
            return [3, 7, 11]
        elif num_mf_layers == 4:
            return [2, 5, 8, 11]
        elif num_mf_layers == 6:
            return [1, 3, 5, 7, 9, 11]
        elif num_mf_layers == 12:
            return list(range(12))
        else:
            raise ValueError(f"Unsupported num_mf_layers: {num_mf_layers}. "
                           f"Supported values are: 1, 2, 3, 4, 6, 12")
    def pick_mf_key(self, i: int) -> str:
        for mf_idx, layer_idx in enumerate(self.mf_layer_indices):
            if i <= layer_idx:
                return f'MF{mf_idx}'
        return f'MF{len(self.mf_layer_indices) - 1}'
    def map_to_mf_coordinates(self, coords, original_resolution, mf_resolution):
        scale = mf_resolution / original_resolution
        return torch.floor(coords * scale)
    def n_linear_interp(self, voxel_embedds, x, voxel_min_vertex, voxel_max_vertex, camera_pe):
        B, N, NN, F = voxel_embedds.shape
        relative_min_dist = (x - voxel_min_vertex) / (voxel_max_vertex - voxel_min_vertex + 1e-12)
        relative_max_dist = (voxel_max_vertex - x) / (voxel_max_vertex - voxel_min_vertex + 1e-12)
        combined_input = torch.cat([x, relative_min_dist, relative_max_dist], dim=-1)
        if self.mode == 'NN':
            weights = self.create_weight(voxel_embedds.reshape(B * N, -1), camera_pe)
            weights = weights.reshape(B, N, -1).unsqueeze(-1)
        else:
            weights = self.create_weight(combined_input, camera_pe).unsqueeze(-1)
        weighted = voxel_embedds * weights
        combined = torch.sum(weighted, dim=2)
        return combined
    def get_voxel_vertices(self, xyz, bounding_box, resolution, log2_hashmap_size):
        device = xyz.device
        D = xyz.size(-1)
        box_min = bounding_box[0].to(device)
        box_max = bounding_box[1].to(device)
        xyz = torch.max(torch.min(xyz, box_max[None, None, :]), box_min[None, None, :])
        grid_size = (box_max - box_min) / resolution
        bottom_left_idx = torch.floor((xyz - box_min) / grid_size).int()
        voxel_min_vertex = bottom_left_idx * grid_size + box_min
        voxel_max_vertex = voxel_min_vertex + grid_size
        BOX_OFFSETS = torch.tensor(list(product([0, 1], repeat=D)), dtype=torch.int32, device=xyz.device)
        expanded_idx = bottom_left_idx.unsqueeze(2)
        total_offsets = BOX_OFFSETS.unsqueeze(0).unsqueeze(0)
        voxel_indices = expanded_idx + total_offsets
        flat_indices = voxel_indices.view(-1, D)
        hashed_voxel_indices = hash(flat_indices, log2_hashmap_size).view(xyz.shape[0], xyz.shape[1], -1)
        return xyz, voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices
    def get_voxel_vertices_coordinates_only(self, xyz, bounding_box, resolution, log2_hashmap_size):
        device = xyz.device
        D = xyz.size(-1)
        box_min = bounding_box[0].to(device)
        box_max = bounding_box[1].to(device)
        xyz = torch.max(torch.min(xyz, box_max[None, None, :]), box_min[None, None, :])
        grid_size = (box_max - box_min) / resolution
        bottom_left_idx = torch.floor((xyz - box_min) / grid_size).int()
        voxel_min_vertex = bottom_left_idx * grid_size + box_min
        voxel_max_vertex = voxel_min_vertex + grid_size
        empty_hash = torch.zeros(xyz.shape[0], xyz.shape[1], 2**D, device=device, dtype=torch.long)
        return xyz, voxel_min_vertex, voxel_max_vertex, empty_hash
    def forward(self, x, i, camera_pe):
        original_resolution = int(self.resolutions[i])
        x_lvl, vmin_lvl, vmax_lvl, _ = self.get_voxel_vertices_coordinates_only(
            x, self.bounding_box, original_resolution, self.log2_hashmap_size
        )
        mf_key = self.pick_mf_key(i)
        mf_resolution = self.mf_resolutions[mf_key]
        x_mf, vmin_mf, vmax_mf, hashed_idx_mf = self.get_voxel_vertices(
            x_lvl, self.bounding_box, mf_resolution, self.log2_hashmap_size
        )
        voxel_embedds = self.mf_embeddings[mf_key](hashed_idx_mf.to(self.device))
        x_embedded = self.n_linear_interp(voxel_embedds, x_mf, vmin_mf, vmax_mf, camera_pe)
        return x_embedded, vmin_mf
    def get_mf_info(self):
        info = {
            'num_mf_layers': self.num_mf_layers,
            'mf_layer_indices': self.mf_layer_indices,
            'mf_resolutions': self.mf_resolutions,
            'out_dim': self.out_dim
        }
        return info
class CameraPatchMLP(nn.Module):
    def __init__(self,
        patch_size,
        camera_pe_size,
        hidden_size,
        output_size,
        embed_features = None,
        num_layers     = 2,
        activation     = 'tanh',
        lr_multiplier  = 1,
        normalize_input = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.camera_pe_size = camera_pe_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.normalize_input = normalize_input
        self.activation = activation
        if embed_features is None:
            embed_features = hidden_size
        self.embed_features = embed_features
        self.camera_embed = FullyConnectedLayer(
            camera_pe_size,
            embed_features,
            activation='tanh',
            lr_multiplier=lr_multiplier
        )
        self.norm_layers = nn.ModuleList()
        if num_layers == 1:
            features_list = [patch_size + embed_features, output_size]
        else:
            features_list = [patch_size + embed_features] + [hidden_size + embed_features] * (num_layers - 2) + [hidden_size + embed_features, output_size]
            for _ in range(num_layers - 1):
                self.norm_layers.append(nn.LayerNorm(hidden_size))
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_features = features_list[i]
            out_features = hidden_size if i < num_layers - 1 else output_size
            layer = FullyConnectedLayer(
                in_features,
                out_features,
                activation='tanh',
                lr_multiplier=lr_multiplier
            )
            self.layers.append(layer)
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'lrelu':
            self.act = nn.LeakyReLU(negative_slope=0.2)
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        else:
            self.act = nn.Identity()
    def forward(self, patches, camera_pe):
        B, num_patches, patch_size = patches.shape
        patches_flat = patches.reshape(B * num_patches, patch_size)
        if self.normalize_input:
            patches_flat = normalize_2nd_moment(patches_flat)
        camera_embed = self.camera_embed(camera_pe)
        if self.normalize_input:
            camera_embed = normalize_2nd_moment(camera_embed)
        camera_embed_expanded = camera_embed.unsqueeze(1).expand(-1, num_patches, -1)
        camera_embed_flat = camera_embed_expanded.reshape(B * num_patches, -1)
        features = torch.cat([patches_flat, camera_embed_flat], dim=1)
        for i, layer in enumerate(self.layers):
            if i == self.num_layers - 1:
                features = layer(features)
            else:
                hidden = layer(features)
                hidden = self.act(hidden)
                if i < len(self.norm_layers):
                    hidden = self.norm_layers[i](hidden)
                features = torch.cat([hidden, camera_embed_flat], dim=1)
        if self.activation == 'tanh':
            output = torch.tanh(features)
        elif self.activation == 'sigmoid':
            output = torch.sigmoid(features)
        else:
            output = features
        output = output.reshape(B, num_patches, self.output_size)
        return output
    def extra_repr(self):
        return f'patch_size={self.patch_size:d}, camera_pe_size={self.camera_pe_size:d}, embed_features={self.embed_features:d}, hidden_size={self.hidden_size:d}, output_size={self.output_size:d}, num_layers={self.num_layers:d}, activation={self.activation:s}'
class PatchEmbedder(nn.Module):
    def __init__(self, image_size, resolutions_list, n_patch_feature, bounding_box, n_levels=16, base_resolution=16, finest_resolution=512,activation='tanh'):
        super(PatchEmbedder, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        C, H, W = image_size
        self.C, self.H, self.W = C, H, W
        self.height = finest_resolution
        self.width = finest_resolution
        self.n_pixel = H*W
        self.bounding_box = bounding_box
        self.n_levels = n_levels
        self.base_resolution = base_resolution
        self.finest_resolution = finest_resolution
        self.b = torch.exp((torch.log(torch.tensor(finest_resolution, dtype=torch.float32)) -
                            torch.log(torch.tensor(base_resolution, dtype=torch.float32))) / (n_levels - 1)).to(self.device)
        self.resolutions = resolutions_list
        self.mlps = nn.ModuleList()
        self.patch_heights = []
        self.patch_widths = []
        for i in range(n_levels):
            resolution = self.resolutions[i]
            patch_height = finest_resolution // resolution
            patch_width = finest_resolution // resolution
            patch_size = C * patch_height * patch_width
            mlp = CameraPatchMLP(
                patch_size=patch_size,
                camera_pe_size=16,
                hidden_size=128,
                output_size=n_patch_feature,
                embed_features=64,
                num_layers=2,
                activation='tanh',
                lr_multiplier=1,
            )
            self.mlps.append(mlp.to(self.device))
        self.activation = activation
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, steps=H),
            torch.linspace(0, 1, steps=W)
        )
        coords = torch.stack((grid_x, grid_y), dim=-1).view(-1, 2)
        self.array_of_coords = coords.to(self.device)
    def forward(self, patches_list, patch_indices):
        results = []
        B = patch_indices[0].shape[0]
        camera_matrices = patches_list[0]['camera_matrix']
        camera_pe = camera_matrices.reshape(camera_matrices.shape[0], -1)
        for i, patch_encoder in enumerate(self.mlps):
            patches = patches_list[i]['patches']
            patch_index = patch_indices[i]
            feature_vectors = patch_encoder(patches, camera_pe)
            if self.activation == 'tanh':
                feature_vectors = torch.tanh(feature_vectors)
            else:
                feature_vectors = torch.sigmoid(feature_vectors)
            selected_features = feature_vectors[torch.arange(B)[:, None], patch_index]
            results.append(selected_features)
        results_tensor = torch.stack(results, dim=0)
        return results_tensor, camera_pe
class EfficientCrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, heads=4, dim_head=None, dropout=0.0):
        super().__init__()
        self.heads = heads
        head_dim = dim_head if dim_head is not None else query_dim // heads
        inner_dim = head_dim * heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)
        self.dropout_p = dropout
    def forward(self, x, context=None):
        context = context if context is not None else x
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        batch_size, query_len, _ = q.shape
        context_len = k.shape[1]
        q = q.reshape(batch_size, query_len, self.heads, -1).transpose(1, 2)
        k = k.reshape(batch_size, context_len, self.heads, -1).transpose(1, 2)
        v = v.reshape(batch_size, context_len, self.heads, -1).transpose(1, 2)
        dropout_p = self.dropout_p if self.training else 0.0
        try:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        except AttributeError:
            scale = (q.shape[-1]) ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = F.softmax(attn, dim=-1)
            if dropout_p > 0:
                attn = F.dropout(attn, p=dropout_p)
            out = attn @ v
        out = out.transpose(1, 2).reshape(batch_size, query_len, -1)
        return self.to_out(out)
class EnhancedFeatureExtractor(nn.Module):
    def __init__(self, input_channels=3, output_channels=32, hidden_channels=64):
        super(EnhancedFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_channels, output_channels, kernel_size=3, padding=1)
        self.skip_connection = nn.Conv2d(input_channels, output_channels, kernel_size=1)
        self.spatial_attention = LightweightSpatialAttention()
        self.relu = nn.ReLU()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.feature_mlp = nn.Sequential(
            nn.Linear(output_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, output_channels)
        )
        self.feature_weighting = nn.Sequential(
            nn.Linear(output_channels, output_channels),
            nn.Sigmoid()
        )
    def forward(self, lr_img):
        B, N, C = lr_img.shape
        H = W = int(math.sqrt(N))
        img = lr_img.permute(0, 2, 1).reshape(B, C, H, W)
        residual = self.skip_connection(img)
        x = self.relu(self.conv1(img))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        x = x + residual
        x = self.relu(x)
        x = self.spatial_attention(x)
        global_features = self.global_pool(x).squeeze(-1).squeeze(-1)
        global_features = self.feature_mlp(global_features)
        pixel_features = x.reshape(B, -1, N).permute(0, 2, 1)
        global_features_expanded = global_features.unsqueeze(1).expand(-1, N, -1)
        weights = self.feature_weighting(global_features_expanded)
        combined_features = pixel_features * weights + global_features_expanded * (1 - weights)
        return combined_features
class LightweightSpatialAttention(nn.Module):
    def __init__(self):
        super(LightweightSpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        attention = torch.sigmoid(attention)
        return x * attention
class CustomMappingNetwork(nn.Module):
    def __init__(self,
        z_dim,
        c_dim,
        w_dim,
        num_layers      = 8,
        embed_features  = None,
        layer_features  = None,
        activation      = 'tanh',
        lr_multiplier   = 0.01,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_layers = num_layers
        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim
        features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]
        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_features)
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)
    def forward(self, z, c):
        if len(z.shape) > 2:
            batch_size = z.shape[0] * z.shape[1]
            z = z.reshape(batch_size, -1)
        if len(c.shape) > 2:
            c = c.reshape(-1, self.c_dim)
        x = normalize_2nd_moment(z.to(torch.float32))
        if self.c_dim > 0:
            y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
            x = torch.cat([x, y], dim=1)
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)
        return x
class TwoStageMappingFusion(nn.Module):
    def __init__(self,
                 feature_dim,
                 camera_pe_dim,
                 lr_img_dim=3,
                 hidden_dim=64,
                 num_layers=4):
        super().__init__()
        self.feature_dim = feature_dim
        self.camera_pe_dim = camera_pe_dim
        self.lr_img_dim = lr_img_dim
        self.img_feature_dim = 32
        self.camera_mapping = CustomMappingNetwork(
            z_dim=feature_dim,
            c_dim=camera_pe_dim,
            w_dim=hidden_dim,
            num_layers=num_layers,
            embed_features=hidden_dim//2,
            activation='tanh'
        )
        self.img_mapping = CustomMappingNetwork(
            z_dim=hidden_dim,
            c_dim=self.img_feature_dim,
            w_dim=feature_dim,
            num_layers=num_layers,
            embed_features=hidden_dim//2,
            activation='tanh'
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def forward(self, x, camera_pe, lr_img, enhanced_feature_extractor):
        B, N, _ = x.shape
        x_flat = x.reshape(B*N, -1)
        camera_pe_expanded = camera_pe.unsqueeze(1).expand(-1, N, -1).reshape(B*N, -1)
        A = self.camera_mapping(x_flat, camera_pe_expanded)
        image_features = enhanced_feature_extractor(lr_img)
        B_out = self.img_mapping(A, image_features)
        output = B_out.reshape(B, N, self.feature_dim)
        return output
class Camera_LR_PreNet(nn.Module):
    def __init__(self, input_dim, camera_pe_size, hidden_dim, output_dim):
        super(Camera_LR_PreNet, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer4 = nn.Linear(hidden_dim, hidden_dim)
        self.layer5 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.norm4 = nn.LayerNorm(hidden_dim)
    def forward(self, x, camera_pe=None):
        B, N, input_dim = x.shape
        x_flat = x.reshape(B*N, -1)
        x1 = self.layer1(x_flat)
        x1 = self.relu(x1)
        x1 = x1.reshape(B, N, -1)
        x1 = self.norm1(x1)
        x1_flat = x1.reshape(B*N, -1)
        x2 = self.layer2(x1_flat)
        x2 = self.relu(x2)
        x2 = x2.reshape(B, N, -1)
        x2 = self.norm2(x2)
        x2_flat = x2.reshape(B*N, -1)
        x3 = self.layer3(x2_flat)
        x3 = self.relu(x3)
        x3 = x3.reshape(B, N, -1)
        x3 = self.norm3(x3)
        x3_flat = x3.reshape(B*N, -1)
        x4 = self.layer4(x3_flat)
        x4 = self.relu(x4)
        x4 = x4.reshape(B, N, -1)
        x4 = self.norm4(x4)
        x4_flat = x4.reshape(B*N, -1)
        x5 = self.layer5(x4_flat)
        output = x5.reshape(B, N, -1)
        return output
class AttentionMLP(nn.Module):
    def __init__(self, input_dim):
        super(AttentionMLP, self).__init__()
        self.attention_mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.attention_mlp(x)
class CameraAwareNetwork(nn.Module):
    def __init__(self, input_dim, camera_pe_size, hidden_dim, output_dim):
        super(CameraAwareNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
    def forward(self, x):
        B, N, input_dim = x.shape
        x_flat = x.reshape(B*N, -1)
        x1 = self.layer1(x_flat)
        x1 = self.relu(x1)
        x1 = x1.reshape(B, N, -1)
        x1 = self.norm1(x1)
        x1_flat = x1.reshape(B*N, -1)
        x2 = self.layer2(x1_flat)
        x2 = self.relu(x2)
        x2 = x2.reshape(B, N, -1)
        x2 = self.norm2(x2)
        x2_flat = x2.reshape(B*N, -1)
        x3 = self.layer3(x2_flat)
        output = torch.sigmoid(x3).reshape(B, N, -1)
        return output
class ImageReconstructionModel(nn.Module):
    def __init__(self, imagesize, resolutions_list, bounding_box, n_levels, n_features_per_level,
                 log2_hashmap_size, base_resolution, finest_resolution,
                 n_dim, mode, n_patch_feature, num_mf_layers, output_dim=3):
        super(ImageReconstructionModel, self).__init__()
        self.hash_embedder = HashEmbedder(resolutions_list, bounding_box, n_levels, n_features_per_level,
                                          log2_hashmap_size, base_resolution, finest_resolution,mode,num_mf_layers)
        self.patch_embedder = PatchEmbedder(imagesize, resolutions_list, n_patch_feature, bounding_box,
                                            n_levels, base_resolution, finest_resolution)
        input_dim = n_dim
        self.C, self.H, self.W = imagesize
        self.learnable_offsets = nn.Parameter(torch.randn(n_levels, n_dim))
        self.attention_mlp = AttentionMLP(n_features_per_level)
        self.learnable_weights_network = CameraAwareNetwork(
            input_dim=n_features_per_level,
            camera_pe_size=16,
            hidden_dim=64,
            output_dim=n_features_per_level
        )
        self.fusion_network = TwoStageMappingFusion(
            feature_dim=n_levels * n_features_per_level,
            camera_pe_dim=16,
            lr_img_dim=3,
            hidden_dim=64,
            num_layers=4
        )
        self.pred_network = Camera_LR_PreNet(
            input_dim=n_levels * n_features_per_level,
            camera_pe_size=16,
            hidden_dim=64,
            output_dim=output_dim
        )
        self.enhanced_feature_extractor = EnhancedFeatureExtractor(
            input_channels=3,
            output_channels=32,
            hidden_channels=64
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_levels = n_levels
        self.base_resolution = base_resolution
        self.finest_resolution = finest_resolution
        self.b = self.patch_embedder.b
        self.n_features_per_level = n_features_per_level
    def decoder(self, z, camera_pe,lr_img):
        feature = self.fusion_network(z, camera_pe, lr_img, self.enhanced_feature_extractor)
        h3 = self.pred_network(feature, camera_pe)
        return torch.sigmoid(h3)
    def forward(self, patch_indices, coords, patches_list, lr_img):
        batch_size = coords.shape[0]
        batch_coords = coords.to(self.device)
        patch_features, camera_pe = self.patch_embedder(patches_list, patch_indices)
        hash_feature = torch.cat((batch_coords.repeat(self.n_levels, 1, 1, 1), patch_features), dim=-1)
        latent_features = []
        for level in range(self.n_levels):
            level_hash_code = hash_feature[level]
            level_latent_features, quant_latent_code = self.hash_embedder(level_hash_code, level, camera_pe)
            latent_features.append(level_latent_features)
        for level in range(1, self.n_levels):
            learnable_weight = self.learnable_weights_network(latent_features[level - 1])
            latent_features[level] += learnable_weight * latent_features[level - 1]
        latent_features_mlp = torch.stack(latent_features, dim=2)
        atten_w = self.attention_mlp(latent_features_mlp)
        weighted_latent_features = latent_features_mlp * atten_w
        final_features = weighted_latent_features.view(batch_size, -1, self.n_levels * self.n_features_per_level)
        lr_img = lr_img.to(self.device)
        reconstructed_img = self.decoder(final_features, camera_pe, lr_img)
        return reconstructed_img
def load_image(image_path, size=None):
    img = Image.open(image_path).convert('RGB')
    if size:
        img = img.resize(size)
    transform = transforms.ToTensor()
    return transform(img).unsqueeze(0)
def display_tensor_image(tensor_img):
    img = vutils.make_grid(tensor_img, normalize=True, scale_each=True)
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.axis('off')
def imshow(img):
    img = img.numpy().transpose((1, 2, 0))
    plt.imshow(img)
    plt.axis('off')
def visualize_batch(data_loader, title):
    images, _ = next(iter(data_loader))
    print(images.shape)
    plt.figure(figsize=(10, 10))
    plt.title(title)
    imshow(vutils.make_grid(images, nrow=4, padding=2, normalize=True))
    plt.show()
def compute_kl_loss(mu, log_var):
    epsilon = 1e-8
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - (log_var.exp() + epsilon))
    return kl_loss
def boosted_criterion(pred, target):
    error = F.mse_loss(pred, target, reduction='none')
    boost_mask = (error < 0.001).float()
    boosted_error = error + boost_mask * 0.05
    return boosted_error.mean()
class ReconLoss(nn.Module):
    def __init__(self, height, width, reduction='sum'):
        super(ReconLoss, self).__init__()
        if reduction=='sum':
            self.mse_loss = nn.MSELoss(reduction='sum')
            self.l1_loss = nn.L1Loss(reduction='sum')
        else:
            self.mse_loss = nn.MSELoss()
            self.l1_loss = nn.L1Loss()
        self.height = height
        self.width = width
    def forward(self, outputs, targets):
        mse = self.mse_loss(outputs, targets)
        return mse
def filter_nan_samples(patch_indices, patch_coords, HR_patches):
    valid_mask = ~torch.isnan(patch_coords).any(dim=-1)
    valid_mask = valid_mask.all(dim=-1)
    for indices in patch_indices:
        valid_mask &= ~torch.isnan(indices).any(dim=-1)
    if not valid_mask.all():
        print(f"Found {(~valid_mask).sum().item()} samples with NaN values in batch of {len(valid_mask)}")
    filtered_patch_coords = patch_coords[valid_mask]
    filtered_HR_patches = HR_patches[valid_mask]
    filtered_patch_indices = [indices[valid_mask] for indices in patch_indices]
    return filtered_patch_indices, filtered_patch_coords, filtered_HR_patches, valid_mask
def calculate_batch_lpips(outputs, targets, device):
    try:
        lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(device)
        if outputs.device != device:
            outputs = outputs.to(device)
        if targets.device != device:
            targets = targets.to(device)
        with torch.no_grad():
            lpips_value = lpips(outputs, targets)
        return lpips_value
    except Exception as e:
        print(f"Error when calculating LPIPS: {str(e)}")
        return torch.tensor(0.0, device=device)
def save_output_images(output_images, file_names, save_dir, epoch=None):
    batch_size = output_images.shape[0]
    if epoch is not None:
        epoch_dir = os.path.join(save_dir, f"epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        save_path_base = epoch_dir
    else:
        save_path_base = save_dir
    if isinstance(file_names, tuple) and len(file_names) == batch_size:
        names_list = list(file_names)
    elif isinstance(file_names, tuple) and len(file_names) != batch_size:
        print(f"({len(file_names)}) mismatch ({batch_size})")
        base_parts = file_names[0].split('_') if file_names else ["img"]
        downscale_factor = base_parts[1] if len(base_parts) > 1 else ""
        names_list = [f"{i:04d}_{downscale_factor}_img_{i}" for i in range(batch_size)]
    elif isinstance(file_names, str):
        names_list = [f"{file_names}_{i}" for i in range(batch_size)]
    else:
        names_list = [f"img_{i}" for i in range(batch_size)]
    for i in range(batch_size):
        img = output_images[i]
        clean_name = names_list[i].replace('/', '_').replace('\\', '_')
        for char in ['(', ')', "'", '"', ',']:
            clean_name = clean_name.replace(char, '')
        clean_name = clean_name.strip()
        save_path = os.path.join(save_path_base, f"{clean_name}.png")
        save_image(img, save_path, normalize=True)
def evaluate_full_dataset(model, data_loader, output_dir, gt_dir, display_name, H,W,epoch=None, log_path=None, max_batches=None, DISPLAY_COL=None):
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    batch_count = 0
    all_outputs = []
    all_targets = []
    with torch.no_grad():
        for data_batches, patch_features, lr_img, _, file_names in tqdm(data_loader, desc=f"Evaluating {display_name} Dataset"):
            if batch_count == 0:
                first_batch_file_names = file_names
            patch_list = [{
                'patches': features['patches'].to(device),
                'camera_matrix': features['camera_matrix'].to(device)
            } for features in patch_features]
            output_hr, gt_hr = [], []
            for batch_data in data_batches:
                batch_patch_indices, batch_coords, batch_hrs, camera_matrix = batch_data
                gt_hr.append(batch_hrs.to(device))
                patch_coords = batch_coords.to(device)
                patch_indices = [indices.to(device) for indices in batch_patch_indices]
                patch_indices, patch_coords, hr_patches, valid_mask = filter_nan_samples(
                    patch_indices, patch_coords, batch_hrs.to(device)
                )
                if len(patch_coords) == 0:
                    print("Ignore this batch because it contains no valid patches...")
                    continue
                outputs = model(patch_indices, patch_coords, patch_list, lr_img.to(device))
                output_hr.append(outputs)
            if len(output_hr) == 0:
                continue
            images = torch.cat(output_hr, dim=1)
            targets = torch.cat(gt_hr, dim=1)
            B, N, C = images.shape
            outputs_reshaped = images.view(B, C, H, W)
            targets_reshaped = targets.view(B, C, H, W)
            save_output_images(outputs_reshaped, file_names, output_dir, epoch)
            if epoch is None or epoch == 1 or not os.listdir(gt_dir):
                save_output_images(targets_reshaped, file_names, gt_dir)
            batch_psnr = peak_signal_noise_ratio(outputs_reshaped, targets_reshaped, data_range=1.0)
            batch_ssim = structural_similarity_index_measure(outputs_reshaped, targets_reshaped, data_range=1.0)
            batch_lpips = calculate_batch_lpips(outputs_reshaped, targets_reshaped, device)
            total_psnr += batch_psnr.item()
            total_ssim += batch_ssim.item()
            total_lpips += batch_lpips.item()
            batch_count += 1
            if batch_count == 1:
                first_batch_outputs = outputs_reshaped.detach().clone()
                first_batch_targets = targets_reshaped.detach().clone()
            if max_batches is not None and batch_count >= max_batches:
                break
    avg_psnr = total_psnr / batch_count if batch_count > 0 else 0
    avg_ssim = total_ssim / batch_count if batch_count > 0 else 0
    avg_lpips = total_lpips / batch_count if batch_count > 0 else 0
    print(f"{display_name} Dataset Evaluation:")
    print(f"Average PSNR: {avg_psnr:.4f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average LPIPS: {avg_lpips:.4f} (lower is better)")
    if log_path:
        with open(log_path, 'a') as f:
            f.write(f"{display_name} Dataset - PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}, LPIPS: {avg_lpips:.4f}")
    if batch_count > 0:
        display_imgs = torch.cat([first_batch_targets[:min(4, len(first_batch_targets))],
                                 first_batch_outputs[:min(4, len(first_batch_outputs))]])
        if epoch is not None:
            vis_dir = os.path.join(output_dir, f"visualizations")
            os.makedirs(vis_dir, exist_ok=True)
            plt.savefig(os.path.join(vis_dir, f"{display_name.lower()}_epoch_{epoch}.png"))
    return avg_psnr, avg_ssim, avg_lpips
def setup_metric_logging(save_dir):
    log_dir = os.path.join(save_dir, 'metrics_logs')
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    train_log_path = os.path.join(log_dir, f'train_metrics_{timestamp}.csv')
    val_log_path = os.path.join(log_dir, f'val_metrics_{timestamp}.csv')
    test_log_path = os.path.join(log_dir, f'test_metrics_{timestamp}.csv')
    headers = ['Epoch', 'Timestamp', 'PSNR', 'SSIM', 'LPIPS']
    for log_path in [train_log_path, val_log_path, test_log_path]:
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    return train_log_path, val_log_path, test_log_path
def log_metrics(log_path, epoch, psnr, ssim, lpips=None):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        psnr_val = float(psnr.item() if hasattr(psnr, 'item') else psnr)
        ssim_val = float(ssim.item() if hasattr(ssim, 'item') else ssim)
        lpips_val = float(lpips.item() if hasattr(lpips, 'item') else (lpips if lpips is not None else 0.0))
    except (ValueError, AttributeError) as e:
        print(f"Eorror - {e}")
        psnr_val = psnr if psnr is not None else 0.0
        ssim_val = ssim if ssim is not None else 0.0
        lpips_val = lpips if lpips is not None else 0.0
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, timestamp, psnr_val, ssim_val, lpips_val])
    print(f"Logged to {log_path}")
def run_model(train_folder, home_dir,dataset,baseres,scale,batch_size,num_mf_layers,
    **kwargs
):

    Train_FOLDER = train_folder
    HOME_DIR = home_dir
    DATASET = dataset
    BASERES = baseres
    SCALE=scale
    num_mf_layers=num_mf_layers

    data_dir = f'{HOME_DIR}/{Train_FOLDER}/'
    SAVE_DIR = f'/project/nianyil/ailab/ICCV2025/wacvmf1/{BASERES}/{SCALE}/{DATASET}/{Train_FOLDER}' 
    OUTPUT_ROOT_DIR = os.path.join(SAVE_DIR, 'generated_images')
    TRAIN_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, 'train')
    TRAIN_GT_DIR = os.path.join(OUTPUT_ROOT_DIR, 'train_GT')
    VAL_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, 'val')
    VAL_GT_DIR = os.path.join(OUTPUT_ROOT_DIR, 'val_GT')
    TEST_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, 'test')
    TEST_GT_DIR = os.path.join(OUTPUT_ROOT_DIR, 'test_GT')

    mode = 'interp'
    activation = 'tanh'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encode = 'TextureEncode'
    
    os.makedirs(TRAIN_OUTPUT_DIR, exist_ok=True)
    os.makedirs(TRAIN_GT_DIR, exist_ok=True)
    os.makedirs(VAL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(VAL_GT_DIR, exist_ok=True)
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEST_GT_DIR, exist_ok=True)
    
    batch_size = batch_size
    num_epochs = 200
    IDX_OFFSET = 0
    coords_dim = 3 
    n_dim = 6
    n_levels = 12
    n_features_per_level = 4
    log2_hashmap_size = 24
    base_resolution = 2 #4
    finest_resolution = BASERES*SCALE
    n_patch_feature =  n_dim - coords_dim

    train_SR_factor = 1
    HR_SCALE = int(train_SR_factor*finest_resolution)

    PIXEL_MAX_N_SCALE = 1 

    downscale_factor_list = [SCALE]

    if activation == 'tanh':
        bounding_box = (-torch.ones(n_dim), torch.ones(n_dim))
    else:
        bounding_box = (torch.zeros(n_dim), torch.ones(n_dim))
    
    transform = transforms.Compose([
        # transforms.Resize(finest_resolution, interpolation=transforms.InterpolationMode.BICUBIC), 
        transforms.Resize((BASERES, BASERES)), 
        transforms.ToTensor()  
    ])
    
    print(f"Loading dataset from {data_dir}")
    train_dataset, val_dataset, test_dataset = get_nerf_datasets(data_dir, transform=transform, finest_resolution=finest_resolution)
    num_train = len(train_dataset)

    lr_image_example, hr_image_example, _ = train_dataset[0]
    C, H, W = 3, finest_resolution, finest_resolution
    #C, H, W = lr_image_example.shape
    # height, width = finest_resolution, finest_resolution
    height, width = H, W
    imagesize = (C, height, width)

    n_pixels = H*W //PIXEL_MAX_N_SCALE 

    n_patches = H // finest_resolution  
    n_patch_HR = H // 128
    if finest_resolution==128:
        total_patches_per_image = n_patches ** 2
    else:
        total_patches_per_image = n_patch_HR** 2

    original_batch_size = batch_size 
    # batch_size = max(1, original_batch_size // total_patches_per_image)
    B = batch_size
    P = n_patches ** 2

    # if num_train == 500:
    #     train_loader = DataLoader(celebahq_full, batch_size=batch_size, shuffle=False)
    #     SAVE_DIR = HOME_DIR + 'output_256/'

    # print("Total images found:", len(train_loader))
    print("Batch size:", batch_size)
    print("Dataset size:", len(train_dataset))

    half_levels = n_levels // 4
    mid_res = finest_resolution // 2
    num_levels_per_resolution = n_levels // 3
    resolutions_list = generate_resolutions_list(base_resolution, finest_resolution, n_levels)
    print(f"Generated resolutions_list: {resolutions_list}")

    
    camera_encoder = CameraPoseEncoder(input_dim=16, hidden_dim=64, output_dim=1)
    camera_encoder = camera_encoder.to(device)
    #viewencoder = ViewDirectionEncoder(input_dim=2, output_dim=1, hidden_dim=64)
    downsampled_train_dataset = DownsampleDatasetWithCoords(
        dataset=train_dataset,
        downscale_factor_list=downscale_factor_list,
        resolutions_list=resolutions_list,
        n_levels=n_levels,
        n_pixels=n_pixels,
        finest_resolution=finest_resolution,
        activation=activation,
        coords_dim=coords_dim,  
        SR_factor=train_SR_factor,
        camera_encoder = camera_encoder
    )

    downsampled_train_loader = DataLoader(downsampled_train_dataset, batch_size=batch_size, shuffle=False)

    first_batch, patch_features, lr_img, img_size, file_names = next(iter(downsampled_train_loader))

    patch_indices, patch_coords, hr_patches, camera_matrix = first_batch[0]

    batch_size = patch_coords.shape[0]
    B = batch_size

    DISPLAY_COL = int(math.sqrt(batch_size))
    DISPLAY_COL = max(DISPLAY_COL, batch_size //DISPLAY_COL)

    print("Shape of the first input image (downsampled and upsampled):", patch_indices[4].shape)
    print("Shape of the corresponding Coords:", patch_coords.shape)
    print("Shape of the corresponding HR images:", hr_patches.shape)
    # print("Shape of the corresponding LR images:", lr_patches.shape)
    print("Original Image Size:", file_names)

    patch_coords = torch.cat([batch[1] for batch in first_batch], dim=1)  
    HR_images = torch.cat([batch[2] for batch in first_batch], dim=1)  
    input_patches = lr_img 

    coords_test=patch_coords
    test_coords = coords_test.view(B, -1, H, W)  

    input_patches = input_patches.view(B, -1, finest_resolution, finest_resolution)
    grid_image = concatenate_images_to_grid(input_patches, DISPLAY_COL)
    print(grid_image.shape)

    HR_images = HR_images.view(B, -1, finest_resolution, finest_resolution)
    hr_grid_image = concatenate_images_to_grid(HR_images, DISPLAY_COL)
    print("HR images grid shape:", hr_grid_image.shape)

    np_image = grid_image.squeeze(0).numpy().transpose(1, 2, 0) 
    plt.figure(figsize=(10, 10)) 
    plt.imshow(np_image)
    plt.axis('off') 
    plt.show()

    np1_image = hr_grid_image.squeeze(0).numpy().transpose(1, 2, 0) 
    plt.figure(figsize=(10, 10)) 
    plt.imshow(np1_image)
    plt.axis('off') 
    plt.show()
    
    downsampled_val_dataset = DownsampleDatasetWithCoords(
        dataset=val_dataset,
        downscale_factor_list=downscale_factor_list,
        resolutions_list=resolutions_list,
        n_levels=n_levels,
        n_pixels=n_pixels,
        finest_resolution=finest_resolution,
        activation=activation,
        coords_dim=coords_dim,
        SR_factor=train_SR_factor,
        camera_encoder=camera_encoder
    )
    
    downsampled_test_dataset = DownsampleDatasetWithCoords(
        dataset=test_dataset,
        downscale_factor_list=downscale_factor_list,
        resolutions_list=resolutions_list,
        n_levels=n_levels,
        n_pixels=n_pixels,
        finest_resolution=finest_resolution,
        activation=activation,
        coords_dim=coords_dim,
        SR_factor=train_SR_factor,
        camera_encoder=camera_encoder
    )

    downsampled_val_loader = DataLoader(downsampled_val_dataset, batch_size=batch_size, shuffle=True)
    downsampled_test_loader = DataLoader(downsampled_test_dataset, batch_size=batch_size, shuffle=True)
    
    model = ImageReconstructionModel(
        imagesize, 
        resolutions_list, 
        bounding_box, 
        n_levels, 
        n_features_per_level,
        log2_hashmap_size, 
        base_resolution, 
        finest_resolution,
        n_dim, 
        mode,
        n_patch_feature,
        num_mf_layers
    )
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.00125)
    use_cuda = torch.cuda.is_available()
    amp_dtype = _amp_dtype()
    scaler = GradScaler(enabled=use_cuda and amp_dtype==torch.float16) 

    criterion = ReconLoss(H, W, reduction='mean')
    creterion_embed = nn.MSELoss(reduction='sum')
    reconstruction_mask = None
    mask_threshold = 0.01
    train_log_path, val_log_path, test_log_path = setup_metric_logging(SAVE_DIR)

    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        
        for data_batches, patch_features, lr_img, _, _ in downsampled_train_loader:
            #print("lr_img.shape",lr_img.shape)
            patch_list = [{
                'patches': features['patches'].to(device),
                'camera_matrix': features['camera_matrix'].to(device)
            } for features in patch_features]
            batch_loss = 0.0

            for batch_data in data_batches:
                batch_patch_indices, batch_coords, batch_hrs, camera_matrix = batch_data

                patch_coords = batch_coords.to(device)
                HR_patches = batch_hrs.to(device)
                patch_indices = [indices.to(device) for indices in batch_patch_indices]
                
                patch_indices, patch_coords, HR_patches, valid_mask = filter_nan_samples(
                patch_indices, patch_coords, HR_patches
                )
            
                
                if len(patch_coords) == 0:
                    print("Entire batch contains NaN, skipping...")
                    continue

                rgb = HR_patches.reshape(HR_patches.size(0), -1, 3)

                optimizer.zero_grad(set_to_none=True)

                with autocast_cm(use_cuda, amp_dtype):
                    outputs = model(patch_indices, patch_coords, patch_list, lr_img)

                    per_pixel_loss = criterion(outputs, rgb)

                    if reconstruction_mask is None:
                        reconstruction_mask = torch.ones_like(per_pixel_loss, device=device)

                    reconstruction_mask = torch.where(per_pixel_loss > mask_threshold, 
                                                reconstruction_mask, 
                                                torch.zeros_like(reconstruction_mask))

                    masked_loss = per_pixel_loss * reconstruction_mask
                    loss_recon = masked_loss.mean()

                    loss_boost = boosted_criterion(outputs, rgb)
                    loss_recon += 5 * loss_boost

                    total_loss = loss_recon
                    
                    batch_loss += total_loss
                
                if amp_dtype == torch.float16 and use_cuda:

                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    optimizer.step()

        avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
        
        if (epoch + 1) % 20 == 0:  
            print(f'Epoch {epoch + 1}, Avg Loss: {avg_epoch_loss:.6f}')
            if device.type == 'cuda':
                print(f"GPU Memory usage: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
            
            eval_log_path = os.path.join(SAVE_DIR, f'evaluation_epoch_{epoch+1}.txt')
            with open(eval_log_path, 'w') as f:
                f.write(f"Evaluation Results for Epoch {epoch+1}\n")
                f.write(f"Average Training Loss: {avg_epoch_loss:.6f}\n")
                f.write("="*50 + "\n")

            max_eval_batches = None  
            
            print("\n" + "="*30 + " Training Set Evaluation " + "="*30)
            train_psnr, train_ssim, train_lpips= evaluate_full_dataset(
                model, downsampled_train_loader, TRAIN_OUTPUT_DIR, TRAIN_GT_DIR, 
                "Training", H,W,epoch+1, eval_log_path, max_eval_batches, DISPLAY_COL
            )
            
            print("\n" + "="*30 + " Validation Set Evaluation " + "="*30)
            val_psnr, val_ssim, val_lpips, = evaluate_full_dataset(
                model, downsampled_val_loader, VAL_OUTPUT_DIR, VAL_GT_DIR, 
                "Validation", H,W,epoch+1, eval_log_path, max_eval_batches, DISPLAY_COL
            )
            
            print("\n" + "="*30 + " Test Set Evaluation " + "="*30)
            test_psnr, test_ssim, test_lpips, = evaluate_full_dataset(
                model, downsampled_test_loader, TEST_OUTPUT_DIR, TEST_GT_DIR, 
                "Test", H,W,epoch+1, eval_log_path, max_eval_batches, DISPLAY_COL
            )
            
            log_metrics(train_log_path, epoch + 1, train_psnr, train_ssim, train_lpips)
            log_metrics(val_log_path, epoch + 1, val_psnr, val_ssim, val_lpips)
            log_metrics(test_log_path, epoch + 1, test_psnr, test_ssim, test_lpips)
            
            print("\n" + "="*30 + f" Epoch {epoch+1} Summary " + "="*30)
            print(f"Training   - PSNR: {train_psnr:.4f}, SSIM: {train_ssim:.4f}, LPIPS: {train_lpips:.4f}")
            print(f"Validation - PSNR: {val_psnr:.4f}, SSIM: {val_ssim:.4f}, LPIPS: {val_lpips:.4f}")
            print(f"Test       - PSNR: {test_psnr:.4f}, SSIM: {test_ssim:.4f}, LPIPS: {test_lpips:.4f}")
            
            if (epoch + 1) % 20 == 0:
                model_save_name = f'{Train_FOLDER}_{IDX_OFFSET}_Image_MultiScale_n_pixels_{n_pixels}_downScale_{downscale_factor_list}_bRes_{base_resolution}_fRes_{finest_resolution}_epoch_{epoch+1}.pth'
                torch.save(model.state_dict(), os.path.join(SAVE_DIR, model_save_name))
        
        model.train()
    final_save_name = f'{Train_FOLDER}_{IDX_OFFSET}_Image_MultiScale_n_pixels_{n_pixels}_downScale_{downscale_factor_list}_bRes_{base_resolution}_fRes_{finest_resolution}_epoch_{num_epochs}.pth'
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, final_save_name))

    print("\n" + "="*30 + " Final Model Evaluation " + "="*30)
    final_eval_log_path = os.path.join(SAVE_DIR, 'final_evaluation.txt')

    print("Evaluating on training set...")
    evaluate_full_dataset(model, downsampled_train_loader, TRAIN_OUTPUT_DIR, TRAIN_GT_DIR, "Training (Final)", H,W,num_epochs, final_eval_log_path, DISPLAY_COL)

    print("Evaluating on validation set...")
    evaluate_full_dataset(model, downsampled_val_loader, VAL_OUTPUT_DIR, VAL_GT_DIR, "Validation (Final)", H,W,num_epochs, final_eval_log_path, DISPLAY_COL)

    print("Evaluating on test set...")
    evaluate_full_dataset(model, downsampled_test_loader, TEST_OUTPUT_DIR, TEST_GT_DIR, "Test (Final)", H,W,num_epochs, final_eval_log_path, DISPLAY_COL)
    
    return model
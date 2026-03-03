import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.utils as vutils


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
import os, json
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import os, json
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
class MultiSceneNeRFDataset(Dataset):
    def __init__(self, root, split='test',
                 train_subfolders=('0','1','2','3','4','5','6'),
                 eval_subfolders=('0','1','2','3','4','5','6'),
                 transform=None, target_transform=None, loader=None):
        self.root = root
        self.split = split
        self.train_subfolders = tuple(train_subfolders)
        self.eval_subfolders = tuple(eval_subfolders)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader if loader is not None else self.default_loader
        self.samples = []
        if split == 'train':
            self._load_train_data()
        else:
            self._load_test_data()
        if len(self.samples) == 0:
            raise RuntimeError(f"[{split}] no samples collected. Check paths/subfolders.")
    def _load_train_data(self):
        base = os.path.join(self.root, 'train')
        if not os.path.exists(base):
            raise RuntimeError(f"Training directory not found: {base}")
        scene_names = sorted([d for d in os.listdir(base)
                              if os.path.isdir(os.path.join(base, d))])
        for scene in scene_names:
            scene_dir = os.path.join(base, scene)
            json_path = os.path.join(scene_dir, 'transforms.json')
            if not os.path.exists(json_path):
                print(f"Warning: Missing transforms.json for scene {scene}")
                continue
            self._append_train_scene(json_path, scene_dir, scene_hint=scene)
    def _load_val_data(self):
        base = os.path.join(self.root, 'test')
        if not os.path.exists(base):
            print(f"Warning: Training directory not found for validation: {base}")
            return
        scene_names = sorted([d for d in os.listdir(base)
                              if os.path.isdir(os.path.join(base, d))])
        for scene in scene_names[:1]:
            scene_dir = os.path.join(base, scene)
            json_path = os.path.join(scene_dir, 'transforms.json')
            if os.path.exists(json_path):
                self._append_train_scene(json_path, scene_dir, scene_hint=f'{scene}(val)')
    def _load_test_data(self):
        base = os.path.join(self.root, 'test')
        if not os.path.exists(base):
            raise RuntimeError(f"Test directory not found: {base}")
        scene_names = sorted([d for d in os.listdir(base)
                              if os.path.isdir(os.path.join(base, d))])
        if not scene_names:
            raise RuntimeError(f"No scene directory under: {base}")
        for scene in scene_names:
            scene_dir = os.path.join(base, scene)
            for sub in self.eval_subfolders:
                sub_dir = os.path.join(scene_dir, sub)
                json_path = os.path.join(sub_dir, 'transforms.json')
                img_root = os.path.join(sub_dir, 'images')
                if os.path.exists(json_path) and os.path.exists(img_root):
                    self._append_test_scene(json_path, img_root, scene_hint=f'{scene}/{sub}')
                else:
                    print(f"Warning: Missing test data for scene {scene}, subfolder {sub}")
    def _append_train_scene(self, json_path, scene_dir, scene_hint=''):
        if not os.path.isfile(json_path):
            print(f"Warning: Missing json: {json_path} (scene {scene_hint})")
            return
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Failed to load json {json_path}: {e}")
            return
        camera_angle_x = data.get('camera_angle_x', None)
        frames = data.get('frames', [])
        added_count = 0
        for fr in frames:
            rel_path = fr.get('file_path', '')
            if not rel_path:
                continue
            if rel_path.startswith('./'):
                rel_path = rel_path[2:]
            img_path = os.path.join(scene_dir, rel_path)
            if not (img_path.endswith('.png') or img_path.endswith('.jpg') or img_path.endswith('.jpeg')):
                img_path = img_path + '.png'
            if os.path.exists(img_path):
                self.samples.append({
                    'img_path': img_path,
                    'transform_matrix': np.array(fr['transform_matrix'], dtype=np.float32),
                    'camera_angle_x': camera_angle_x,
                    'file_path': fr.get('file_path', ''),
                    'scene_name': scene_hint
                })
                added_count += 1
            else:
                print(f"Warning: Image not found: {img_path}")
        if added_count > 0:
            print(f"Added {added_count} samples from scene {scene_hint}")
        else:
            print(f"Warning: No valid samples found in scene {scene_hint}")
    def _append_test_scene(self, json_path, image_root, scene_hint=''):
        if not os.path.isfile(json_path):
            print(f"Warning: Missing json: {json_path} (scene {scene_hint})")
            return
        if not os.path.isdir(image_root):
            print(f"Warning: Missing images folder: {image_root} (scene {scene_hint})")
            return
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Failed to load json {json_path}: {e}")
            return
        camera_angle_x = data.get('camera_angle_x', None)
        frames = data.get('frames', [])
        added_count = 0
        for fr in frames:
            rel = fr.get('file_path', '')
            if rel.startswith('./'):
                rel = rel[2:]
            if rel.startswith('images/'):
                rel = rel[len('images/'):]
            img_path = os.path.join(image_root, rel)
            if not (img_path.endswith('.png') or img_path.endswith('.jpg') or img_path.endswith('.jpeg')):
                img_path = img_path + '.png'
            if os.path.exists(img_path):
                self.samples.append({
                    'img_path': img_path,
                    'transform_matrix': np.array(fr['transform_matrix'], dtype=np.float32),
                    'camera_angle_x': camera_angle_x,
                    'file_path': fr.get('file_path', ''),
                    'scene_name': scene_hint
                })
                added_count += 1
            else:
                print(f"Warning: Image not found: {img_path}")
        if added_count > 0:
            print(f"Added {added_count} samples from {scene_hint}")
        else:
            print(f"Warning: No valid samples found in {scene_hint}")
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        item = self.samples[index]
        img = self.loader(item['img_path'])
        if self.transform is not None:
            img = self.transform(img)
        camera_data = {
            'transform_matrix': item['transform_matrix'],
            'file_path': item['file_path'],
        }
        if item['camera_angle_x'] is not None:
            camera_data['camera_angle_x'] = item['camera_angle_x']
        if 'scene_name' in item:
            camera_data['scene_name'] = item['scene_name']
        if self.target_transform is not None:
            camera_data = self.target_transform(camera_data)
        return img, camera_data
    @staticmethod
    def default_loader(path):
        img = Image.open(path)
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        bg = Image.new('RGB', img.size, (255, 255, 255))
        if img.mode == 'RGBA':
            bg.paste(img, mask=img.split()[3])
            img = bg
        else:
            img = img.convert('RGB')
        return img
class DTUNeRFDatasetBlender(Dataset):
    def __init__(self, root, split='test',
                 train_subfolders=('0','1','2','3','4','5','6'),
                 eval_subfolders=('0','1','2','3','4','5','6'),
                 transform=None, target_transform=None, loader=None, finest_resolution=None):
        self.root = root
        self.split = split
        self.train_subfolders = tuple(train_subfolders)
        self.eval_subfolders = tuple(eval_subfolders)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader if loader is not None else self.default_loader
        self.finest_resolution = finest_resolution
        self.samples = []
        if split == 'train':
            self._load_train_data()
        elif split == 'val':
            self._load_val_data()
        else:
            self._load_test_data()
        if len(self.samples) == 0:
            raise RuntimeError(f"[{split}] no samples collected. Check paths/subfolders.")
    def _load_train_data(self):
        base = os.path.join(self.root, 'train')
        if not os.path.exists(base):
            raise RuntimeError(f"Training directory not found: {base}")
        scene_names = sorted([d for d in os.listdir(base)
                              if os.path.isdir(os.path.join(base, d))])
        for scene in scene_names:
            scene_dir = os.path.join(base, scene)
            json_path = os.path.join(scene_dir, 'transforms.json')
            if not os.path.exists(json_path):
                print(f"Warning: Missing transforms.json for scene {scene}")
                continue
            self._append_train_scene(json_path, scene_dir, scene_hint=scene)
    def _load_val_data(self):
        base = os.path.join(self.root, 'test')
        if not os.path.exists(base):
            print(f"Warning: Test directory not found for validation: {base}")
            return
        scene_names = sorted([d for d in os.listdir(base)
                              if os.path.isdir(os.path.join(base, d))])
        for scene in scene_names[:1]:
            scene_dir = os.path.join(base, scene)
            json_path = os.path.join(scene_dir, 'transforms.json')
            if os.path.exists(json_path):
                self._append_train_scene(json_path, scene_dir, scene_hint=f'{scene}(val)')
    def _load_test_data(self):
        base = os.path.join(self.root, 'test')
        if not os.path.exists(base):
            raise RuntimeError(f"Test directory not found: {base}")
        scene_names = sorted([d for d in os.listdir(base)
                              if os.path.isdir(os.path.join(base, d))])
        if not scene_names:
            raise RuntimeError(f"No scene directory under: {base}")
        for scene in scene_names:
            scene_dir = os.path.join(base, scene)
            for sub in self.eval_subfolders:
                sub_dir = os.path.join(scene_dir, sub)
                json_path = os.path.join(sub_dir, 'transforms.json')
                img_root = os.path.join(sub_dir, 'images')
                if os.path.exists(json_path) and os.path.exists(img_root):
                    self._append_test_scene(json_path, img_root, scene_hint=f'{scene}/{sub}')
                else:
                    print(f"Warning: Missing test data for scene {scene}, subfolder {sub}")
    def _append_train_scene(self, json_path, scene_dir, scene_hint=''):
        if not os.path.isfile(json_path):
            print(f"Warning: Missing json: {json_path} (scene {scene_hint})")
            return
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Failed to load json {json_path}: {e}")
            return
        camera_angle_x = data.get('camera_angle_x', None)
        frames = data.get('frames', [])
        added_count = 0
        for fr in frames:
            rel_path = fr.get('file_path', '')
            if not rel_path:
                continue
            if rel_path.startswith('./'):
                rel_path = rel_path[2:]
            img_path = os.path.join(scene_dir, rel_path)
            if not (img_path.endswith('.png') or img_path.endswith('.jpg') or img_path.endswith('.jpeg')):
                img_path = img_path + '.png'
            if os.path.exists(img_path):
                self.samples.append({
                    'img_path': img_path,
                    'transform_matrix': np.array(fr['transform_matrix'], dtype=np.float32),
                    'camera_angle_x': camera_angle_x,
                    'file_path': fr.get('file_path', ''),
                    'scene_name': scene_hint
                })
                added_count += 1
            else:
                print(f"Warning: Image not found: {img_path}")
        if added_count > 0:
            print(f"Added {added_count} samples from scene {scene_hint}")
        else:
            print(f"Warning: No valid samples found in scene {scene_hint}")
    def _append_test_scene(self, json_path, image_root, scene_hint=''):
        if not os.path.isfile(json_path):
            print(f"Warning: Missing json: {json_path} (scene {scene_hint})")
            return
        if not os.path.isdir(image_root):
            print(f"Warning: Missing images folder: {image_root} (scene {scene_hint})")
            return
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Failed to load json {json_path}: {e}")
            return
        camera_angle_x = data.get('camera_angle_x', None)
        frames = data.get('frames', [])
        added_count = 0
        for fr in frames:
            rel = fr.get('file_path', '')
            if rel.startswith('./'):
                rel = rel[2:]
            if rel.startswith('images/'):
                rel = rel[len('images/'):]
            img_path = os.path.join(image_root, rel)
            if not (img_path.endswith('.png') or img_path.endswith('.jpg') or img_path.endswith('.jpeg')):
                img_path = img_path + '.png'
            if os.path.exists(img_path):
                self.samples.append({
                    'img_path': img_path,
                    'transform_matrix': np.array(fr['transform_matrix'], dtype=np.float32),
                    'camera_angle_x': camera_angle_x,
                    'file_path': fr.get('file_path', ''),
                    'scene_name': scene_hint
                })
                added_count += 1
            else:
                print(f"Warning: Image not found: {img_path}")
        if added_count > 0:
            print(f"Added {added_count} samples from {scene_hint}")
        else:
            print(f"Warning: No valid samples found in {scene_hint}")
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        item = self.samples[index]
        img = self.loader(item['img_path'])
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        camera_data = {
            'transform_matrix': item['transform_matrix'],
            'file_path': item['file_path']
        }
        if item['camera_angle_x'] is not None:
            camera_data['camera_angle_x'] = item['camera_angle_x']
        if 'scene_name' in item:
            camera_data['scene_name'] = item['scene_name']
        if self.target_transform is not None:
            camera_data = self.target_transform(camera_data)
        return img, camera_data
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
def get_dtu_datasets(root, transform=None, target_transform=None, finest_resolution=None):
    all_subfolders = ('0', '1', '2', '3', '4', '5', '6')
    train_dataset = DTUNeRFDatasetBlender(
        root, 'train',
        train_subfolders=all_subfolders,
        eval_subfolders=all_subfolders,
        transform=transform, target_transform=target_transform,
        finest_resolution=finest_resolution
    )
    val_dataset = DTUNeRFDatasetBlender(
        root, 'train',
        train_subfolders=all_subfolders,
        eval_subfolders=all_subfolders,
        transform=transform, target_transform=target_transform,
        finest_resolution=finest_resolution
    )
    if len(val_dataset) > 0:
        val_subset_size = max(1, len(val_dataset) // 5)
        val_dataset.samples = val_dataset.samples[:val_subset_size]
        print(f"Using {len(val_dataset.samples)} samples for validation (subset of training data)")
    try:
        test_dataset = DTUNeRFDatasetBlender(
            root, 'test',
            train_subfolders=all_subfolders,
            eval_subfolders=all_subfolders,
            transform=transform, target_transform=target_transform,
            finest_resolution=finest_resolution
        )
        print(f"Loaded {len(test_dataset)} samples for testing")
    except RuntimeError as e:
        print(f"Test dataset loading failed: {e}")
        print("Using train dataset for testing as fallback")
        test_dataset = DTUNeRFDatasetBlender(
            root, 'train',
            train_subfolders=all_subfolders,
            eval_subfolders=all_subfolders,
            transform=transform, target_transform=target_transform,
            finest_resolution=finest_resolution
        )
        if len(test_dataset) > 0:
            test_subset_start = len(test_dataset) - max(1, len(test_dataset) // 5)
            test_dataset.samples = test_dataset.samples[test_subset_start:]
            print(f"Using {len(test_dataset.samples)} samples for testing (subset of training data)")
    return train_dataset, val_dataset, test_dataset
def get_nerf_datasets(root, transform=None, target_transform=None):
    all_subfolders = ('0', '1', '2', '3', '4', '5', '6')
    train_dataset = MultiSceneNeRFDataset(
        root, 'train',
        train_subfolders=all_subfolders,
        eval_subfolders=all_subfolders,
        transform=transform, target_transform=target_transform
    )
    val_dataset = MultiSceneNeRFDataset(
        root, 'val',
        train_subfolders=all_subfolders,
        eval_subfolders=all_subfolders,
        transform=transform, target_transform=target_transform
    )
    test_dataset = MultiSceneNeRFDataset(
        root, 'test',
        train_subfolders=all_subfolders,
        eval_subfolders=all_subfolders,
        transform=transform, target_transform=target_transform
    )
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

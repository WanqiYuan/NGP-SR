import csv
import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision import transforms
from torchvision.utils import save_image
import torchvision.utils as vutils
from tqdm import tqdm

from embedders import (
    AttentionMLP,
    CameraAwareNetwork,
    Camera_LR_PreNet,
    EnhancedFeatureExtractor,
    HashEmbedder,
    PatchEmbedder,
    TwoStageMappingFusion,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

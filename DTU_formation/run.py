import glob
import math
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from tqdm import tqdm

from model import (
    ImageReconstructionModel,
    ReconLoss,
    boosted_criterion,
    calculate_batch_lpips,
    evaluate_full_dataset,
    filter_nan_samples,
    log_metrics,
    save_output_images,
    setup_metric_logging,
)
from utilities import (
    CameraPoseEncoder,
    DownsampleDatasetWithCoords,
    _amp_dtype,
    autocast_cm,
    concatenate_images_to_grid,
    generate_resolutions_list,
    get_dtu_datasets,
)
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
def cleanup():
    dist.destroy_process_group()
def save_checkpoint(model, optimizer, scaler, epoch, loss, save_dir, train_folder, idx_offset,
                   n_pixels, downscale_factor_list, base_resolution, finest_resolution, rank):
    if rank == 0:
        model_to_save = model.module if hasattr(model, 'module') else model
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'loss': loss,
            'train_folder': train_folder,
            'idx_offset': idx_offset,
            'n_pixels': n_pixels,
            'downscale_factor_list': downscale_factor_list,
            'base_resolution': base_resolution,
            'finest_resolution': finest_resolution
        }
        checkpoint_name = f'{train_folder}_{idx_offset}_checkpoint_epoch_{epoch}.pth'
        checkpoint_path = os.path.join(save_dir, checkpoint_name)
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        latest_checkpoint_path = os.path.join(save_dir, 'latest_checkpoint.txt')
        with open(latest_checkpoint_path, 'w') as f:
            f.write(checkpoint_name)
        try:
            import glob, re
            pattern = os.path.join(save_dir, f'{train_folder}_{idx_offset}_checkpoint_epoch_*.pth')
            checkpoint_files = glob.glob(pattern)
            def extract_epoch(path):
                m = re.search(r'epoch_(\d+)\.pth$', path)
                return int(m.group(1)) if m else -1
            checkpoint_files.sort(key=extract_epoch, reverse=True)
            to_delete = checkpoint_files[2:]
            for old_path in to_delete:
                try:
                    os.remove(old_path)
                    print(f"Removed old checkpoint: {old_path}")
                except Exception as e:
                    print(f"Failed to remove old checkpoint {old_path}: {e}")
        except Exception as e:
            print(f"Checkpoint cleanup skipped due to error: {e}")
def load_checkpoint(checkpoint_path, model, optimizer, scaler, rank):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if rank == 0:
        print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{rank}')
    model_to_load = model.module if hasattr(model, 'module') else model
    model_to_load.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    loss = checkpoint['loss']
    if rank == 0:
        print(f"Resumed from epoch {checkpoint['epoch']}, starting epoch {start_epoch}")
        print(f"Last recorded loss: {loss:.6f}")
    return start_epoch, loss
def find_latest_checkpoint(checkpoint_dir, train_folder, idx_offset):
    if not os.path.exists(checkpoint_dir):
        return None
    latest_checkpoint_file = os.path.join(checkpoint_dir, 'latest_checkpoint.txt')
    if os.path.exists(latest_checkpoint_file):
        with open(latest_checkpoint_file, 'r') as f:
            checkpoint_name = f.read().strip()
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            if os.path.exists(checkpoint_path):
                return checkpoint_path
    pattern = f'{train_folder}_{idx_offset}_checkpoint_epoch_*.pth'
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, pattern))
    if not checkpoint_files:
        return None
    def extract_epoch(filename):
        match = re.search(r'epoch_(\d+)\.pth$', filename)
        return int(match.group(1)) if match else -1
    latest_checkpoint = max(checkpoint_files, key=extract_epoch)
    return latest_checkpoint
def create_distributed_datasets(train_dataset, val_dataset, test_dataset, downscale_factor_list,
                               resolutions_list, n_levels, n_pixels, finest_resolution,
                               activation, coords_dim, train_SR_factor, camera_encoder, rank, world_size):
    camera_encoder_cpu = CameraPoseEncoder(input_dim=16, hidden_dim=64, output_dim=1)
    camera_encoder_cpu.load_state_dict(camera_encoder.state_dict())
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
        camera_encoder=camera_encoder_cpu
    )
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
        camera_encoder=camera_encoder_cpu
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
        camera_encoder=camera_encoder_cpu
    )
    train_sampler = DistributedSampler(
        downsampled_train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    val_sampler = DistributedSampler(
        downsampled_val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    test_sampler = DistributedSampler(
        downsampled_test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    return (downsampled_train_dataset, downsampled_val_dataset, downsampled_test_dataset,
            train_sampler, val_sampler, test_sampler)
def evaluate_full_dataset_single_gpu(model, data_loader, output_dir, gt_dir, display_name, H, W,
                                   epoch=None, log_path=None, max_batches=None, DISPLAY_COL=None):
    model.eval()
    device = next(model.parameters()).device
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    batch_count = 0
    with torch.no_grad():
        for data_batches, patch_features, lr_img, _, file_names in tqdm(data_loader, desc=f"Evaluating {display_name} Dataset"):
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
            if max_batches is not None and batch_count >= max_batches:
                break
    avg_psnr = total_psnr / batch_count if batch_count > 0 else 0
    avg_ssim = total_ssim / batch_count if batch_count > 0 else 0
    avg_lpips = total_lpips / batch_count if batch_count > 0 else 0
    print(f"{display_name} Dataset Evaluation:")
    print(f"Average PSNR: {avg_psnr:.4f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average LPIPS: {avg_lpips:.4f} (lower is better)")
    return avg_psnr, avg_ssim, avg_lpips
def run_model_ddp(rank, world_size, train_folder, home_dir, dataset, baseres, scale, batch_size,
                  resume=False, checkpoint_dir=None, checkpoint_path=None, **kwargs):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    Train_FOLDER = train_folder
    HOME_DIR = home_dir
    DATASET = dataset
    BASERES = baseres
    SCALE = scale
    data_dir = f'{HOME_DIR}/'
    SAVE_DIR = f'/project/nianyil/ailab/ICCV2025/dtus_mf1_retrain3singlefake/{BASERES}/{SCALE}/{DATASET}'
    if rank == 0:
        os.makedirs(SAVE_DIR, exist_ok=True)
    dist.barrier()
    mode = 'interp'
    activation = 'tanh'
    encode = 'TextureEncode'
    effective_batch_size = max(1, batch_size // world_size)
    if batch_size < world_size:
        if rank == 0:
            print(f"Warning: batch_size ({batch_size}) is smaller than world_size ({world_size})")
            print(f"Setting effective_batch_size to 1 for each GPU")
        effective_batch_size = 1
    if rank == 0:
        print(f"Original batch_size: {batch_size}")
        print(f"World_size: {world_size}")
        print(f"Effective batch_size per GPU: {effective_batch_size}")
        print(f"Total effective batch_size: {effective_batch_size * world_size}")
    num_epochs = 200
    IDX_OFFSET = 0
    coords_dim = 3
    n_dim = 6
    n_levels = 12
    n_features_per_level = 4
    log2_hashmap_size = 24
    base_resolution = 2
    finest_resolution = BASERES * SCALE
    n_patch_feature = n_dim - coords_dim
    train_SR_factor = 1
    HR_SCALE = int(train_SR_factor * finest_resolution)
    PIXEL_MAX_N_SCALE = 1
    downscale_factor_list = [SCALE]
    if activation == 'tanh':
        bounding_box = (-torch.ones(n_dim), torch.ones(n_dim))
    else:
        bounding_box = (torch.zeros(n_dim), torch.ones(n_dim))
    transform = transforms.Compose([
        transforms.Resize((HR_SCALE, HR_SCALE)),
        transforms.ToTensor()
    ])
    if rank == 0:
        print(f"Loading dataset from {data_dir}")
    train_dataset, val_dataset, test_dataset = get_dtu_datasets(data_dir, transform=transform, finest_resolution=finest_resolution)
    image_example, _ = train_dataset[0]
    C, H, W = image_example.shape
    height, width = H, W
    imagesize = (C, height, width)
    n_pixels = H * W // PIXEL_MAX_N_SCALE
    resolutions_list = generate_resolutions_list(base_resolution, finest_resolution, n_levels)
    camera_encoder = CameraPoseEncoder(input_dim=16, hidden_dim=64, output_dim=1)
    camera_encoder = camera_encoder.to(device)
    (downsampled_train_dataset, downsampled_val_dataset, downsampled_test_dataset,
     train_sampler, val_sampler, test_sampler) = create_distributed_datasets(
        train_dataset, val_dataset, test_dataset, downscale_factor_list,
        resolutions_list, n_levels, n_pixels, finest_resolution,
        activation, coords_dim, train_SR_factor, camera_encoder, rank, world_size
    )
    num_workers = 0 if effective_batch_size == 1 else min(2, effective_batch_size)
    downsampled_train_loader = DataLoader(
        downsampled_train_dataset,
        batch_size=effective_batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
        persistent_workers=False if num_workers == 0 else True
    )
    num_mf_layers = 1
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
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    base_lr = 0.00125
    reference_batch_size = 8
    effective_total_batch_size = effective_batch_size * world_size
    adjusted_lr = base_lr * (effective_total_batch_size / reference_batch_size)
    if rank == 0:
        print(f"Base learning rate: {base_lr}")
        print(f"Adjusted learning rate: {adjusted_lr}")
    optimizer = optim.Adam(model.parameters(), lr=adjusted_lr)
    criterion = ReconLoss(H, W, reduction='mean')
    use_cuda = torch.cuda.is_available()
    amp_dtype = _amp_dtype()
    scaler = GradScaler(enabled=use_cuda and amp_dtype==torch.float16)
    reconstruction_mask = None
    mask_threshold = 0.01
    start_epoch = 0
    if resume or checkpoint_path is not None:
        actual_checkpoint_path = None
        if checkpoint_path is not None:
            actual_checkpoint_path = checkpoint_path
            if rank == 0:
                print(f"Using specified checkpoint: {checkpoint_path}")
        else:
            search_dir = checkpoint_dir if checkpoint_dir is not None else SAVE_DIR
            actual_checkpoint_path = find_latest_checkpoint(search_dir, Train_FOLDER, IDX_OFFSET)
            if actual_checkpoint_path is None:
                if rank == 0:
                    print(f"No checkpoint found in {search_dir}, starting from scratch")
            else:
                if rank == 0:
                    print(f"Found latest checkpoint: {actual_checkpoint_path}")
        if actual_checkpoint_path is not None:
            try:
                start_epoch, last_loss = load_checkpoint(
                    actual_checkpoint_path, model, optimizer, scaler, rank
                )
                if rank == 0:
                    print(f"Successfully resumed from epoch {start_epoch-1}")
                    print(f"Will start training from epoch {start_epoch}")
            except Exception as e:
                if rank == 0:
                    print(f"Failed to load checkpoint: {e}")
                    print("Starting training from scratch")
                start_epoch = 0
    if rank == 0 and start_epoch > 0:
        print(f"Resuming training from epoch {start_epoch} to {num_epochs}")
    elif rank == 0:
        print(f"Starting training from epoch 1 to {num_epochs}")
    for epoch in range(start_epoch, num_epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        if rank == 0:
            train_bar = tqdm(downsampled_train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        else:
            train_bar = downsampled_train_loader
        try:
            for data_batches, patch_features, lr_img, _, _ in train_bar:
                try:
                    patch_list = [{
                        'patches': features['patches'].to(device, non_blocking=True),
                        'camera_matrix': features['camera_matrix'].to(device, non_blocking=True)
                    } for features in patch_features]
                    batch_loss = 0.0
                    for batch_data in data_batches:
                        batch_patch_indices, batch_coords, batch_hrs, camera_matrix = batch_data
                        patch_coords = batch_coords.to(device, non_blocking=True)
                        HR_patches = batch_hrs.to(device, non_blocking=True)
                        patch_indices = [indices.to(device, non_blocking=True) for indices in batch_patch_indices]
                        patch_indices, patch_coords, HR_patches, valid_mask = filter_nan_samples(
                            patch_indices, patch_coords, HR_patches
                        )
                        if len(patch_coords) == 0:
                            if rank == 0:
                                print("Entire batch contains NaN, skipping...")
                            continue
                        rgb = HR_patches.reshape(HR_patches.size(0), -1, 3)
                        optimizer.zero_grad(set_to_none=True)
                        with autocast_cm(use_cuda, amp_dtype):
                            outputs = model(patch_indices, patch_coords, patch_list, lr_img.to(device, non_blocking=True))
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
                except RuntimeError as e:
                    if rank == 0:
                        print(f"Error processing batch: {e}")
                    continue
                epoch_loss += batch_loss.item() if hasattr(batch_loss, 'item') else batch_loss
                batch_count += 1
        except Exception as e:
            if rank == 0:
                print(f"Error in training loop: {e}")
            break
        avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
        if (epoch + 1) % 10 == 0:
            if rank == 0:
                print(f'Epoch {epoch + 1}, Avg Loss: {avg_epoch_loss:.6f}')
                if device.type == 'cuda':
                    print(f"GPU Memory usage: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
                save_checkpoint(
                    model, optimizer, scaler, epoch, avg_epoch_loss, SAVE_DIR,
                    Train_FOLDER, IDX_OFFSET, n_pixels, downscale_factor_list,
                    base_resolution, finest_resolution, rank
                )
                model_save_name = f'{Train_FOLDER}_{IDX_OFFSET}_Image_MultiScale_n_pixels_{n_pixels}_downScale_{downscale_factor_list}_bRes_{base_resolution}_fRes_{finest_resolution}_epoch_{epoch+1}.pth'
                torch.save(model.module.state_dict(), os.path.join(SAVE_DIR, model_save_name))
                print(f"Model saved at epoch {epoch+1}")
            dist.barrier()
        dist.barrier()
    if rank == 0:
        save_checkpoint(
            model, optimizer, scaler, num_epochs-1, 0.0, SAVE_DIR,
            Train_FOLDER, IDX_OFFSET, n_pixels, downscale_factor_list,
            base_resolution, finest_resolution, rank
        )
        final_save_name = f'{Train_FOLDER}_{IDX_OFFSET}_Image_MultiScale_n_pixels_{n_pixels}_downScale_{downscale_factor_list}_bRes_{base_resolution}_fRes_{finest_resolution}_epoch_{num_epochs}.pth'
        torch.save(model.module.state_dict(), os.path.join(SAVE_DIR, final_save_name))
        print(f"Final model saved after {num_epochs} epochs")
    dist.barrier()
    cleanup()
    return model.module if rank == 0 else None
def run_model(train_folder, home_dir,dataset,baseres,scale,batch_size,
    **kwargs
):
    Train_FOLDER = train_folder
    HOME_DIR = home_dir
    DATASET = dataset
    BASERES = baseres
    SCALE=scale
    data_dir = f'{HOME_DIR}/'
    SAVE_DIR = f'/project/nianyil/ailab/ICCV2025/dtus_mf1_retrain3singleload/{BASERES}/{SCALE}/{DATASET}'
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
    base_resolution = 2
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
        transforms.Resize((HR_SCALE, HR_SCALE)),
        transforms.ToTensor()
    ])
    print(f"Loading dataset from {data_dir}")
    train_dataset, val_dataset, test_dataset = get_dtu_datasets(data_dir, transform=transform, finest_resolution=finest_resolution)
    num_train = len(train_dataset)
    image_example, _ = train_dataset[0]
    C, H, W = image_example.shape
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
    B = batch_size
    P = n_patches ** 2
    print("Batch size:", batch_size)
    print("Dataset size:", len(train_dataset))
    half_levels = n_levels // 4
    mid_res = finest_resolution // 2
    num_levels_per_resolution = n_levels // 3
    resolutions_list = generate_resolutions_list(base_resolution, finest_resolution, n_levels)
    print(f"Generated resolutions_list: {resolutions_list}")
    camera_encoder = CameraPoseEncoder(input_dim=16, hidden_dim=64, output_dim=1)
    camera_encoder = camera_encoder.to(device)
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
    num_mf_layers=1
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
    start_epoch = 0
    resume = kwargs.get('resume', False)
    checkpoint_dir = kwargs.get('checkpoint_dir', None)
    checkpoint_path = kwargs.get('checkpoint_path', None)
    if resume or checkpoint_path is not None:
        actual_checkpoint_path = None
        if checkpoint_path is not None:
            actual_checkpoint_path = checkpoint_path
            print(f"Using specified checkpoint: {checkpoint_path}")
        else:
            search_dir = checkpoint_dir if checkpoint_dir is not None else SAVE_DIR
            actual_checkpoint_path = find_latest_checkpoint(search_dir, Train_FOLDER, IDX_OFFSET)
            if actual_checkpoint_path is None:
                print(f"No checkpoint found in {search_dir}, starting from scratch")
            else:
                print(f"Found latest checkpoint: {actual_checkpoint_path}")
        if actual_checkpoint_path is not None:
            try:
                start_epoch, last_loss = load_checkpoint(
                    actual_checkpoint_path, model, optimizer, scaler, rank=0
                )
                print(f"Successfully resumed from epoch {start_epoch-1}")
                print(f"Will start training from epoch {start_epoch}")
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")
                print("Starting training from scratch")
                start_epoch = 0
    if start_epoch > 0:
        print(f"Resuming training from epoch {start_epoch} to {num_epochs}")
    else:
        print(f"Starting training from epoch 1 to {num_epochs}")
    for epoch in tqdm(range(start_epoch, num_epochs), desc="Training Progress"):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        for data_batches, patch_features, lr_img, _, _ in downsampled_train_loader:
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
                save_checkpoint(
                    model, optimizer, scaler, epoch, avg_epoch_loss, SAVE_DIR,
                    Train_FOLDER, IDX_OFFSET, n_pixels, downscale_factor_list,
                    base_resolution, finest_resolution, rank=0
                )
                model_save_name = f'{Train_FOLDER}_{IDX_OFFSET}_Image_MultiScale_n_pixels_{n_pixels}_downScale_{downscale_factor_list}_bRes_{base_resolution}_fRes_{finest_resolution}_epoch_{epoch+1}.pth'
                torch.save(model.state_dict(), os.path.join(SAVE_DIR, model_save_name))
        model.train()
    save_checkpoint(
        model, optimizer, scaler, num_epochs-1, 0.0, SAVE_DIR,
        Train_FOLDER, IDX_OFFSET, n_pixels, downscale_factor_list,
        base_resolution, finest_resolution, rank=0
    )
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

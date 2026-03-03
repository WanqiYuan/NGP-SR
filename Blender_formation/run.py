import math
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from model import (
    ImageReconstructionModel,
    ReconLoss,
    boosted_criterion,
    evaluate_full_dataset,
    log_metrics,
    setup_metric_logging,
    filter_nan_samples,
)
from utilities import (
    CameraPoseEncoder,
    DownsampleDatasetWithCoords,
    _amp_dtype,
    autocast_cm,
    concatenate_images_to_grid,
    generate_resolutions_list,
    get_nerf_datasets,
)
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
    # SAVE_DIR = HOME_DIR + 'output_500/'
    SAVE_DIR = f'/project/nianyil/ailab/ICCV2025/testgithub/{BASERES}/{SCALE}/{DATASET}/{Train_FOLDER}' 
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
    finest_resolution = BASERES*SCALE#256 
    n_patch_feature =  n_dim - coords_dim
    # n_latent_feature = 10

    train_SR_factor = 1
    HR_SCALE = int(train_SR_factor*finest_resolution)

    PIXEL_MAX_N_SCALE = 1 # n_pixels = H*W//PIXEL_MAX_N_SCALE

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
    # Initialize DownsampleDatasetWithCoords with all downscale factors
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
                    
                    #total_loss = loss_recon + diffusion_loss
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

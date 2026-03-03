import argparse
import os
import torch
import torch.multiprocessing as mp
import run_dtu
def parse_arguments():
    #Parse command line arguments with default values from the parameter file.
    parser = argparse.ArgumentParser(description='Neural Rendering Model Training and Evaluation')
    parser.add_argument('--train_folder', type=str, default="DTU", help='Training folder name')
    parser.add_argument('--home_dir', type=str, default="/project/nianyil/ailab/NGP_SR_results/data/DTU_512",
                        help='Home directory path')
    parser.add_argument('--dataset', type=str, default="DTU", help='Dataset name')
    parser.add_argument('--baseres', type=int, default=128, help='Base resolution')
    parser.add_argument('--scale', type=int, default=4, help='Scale factor')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--gpus', type=str, default=None,
                        help='Specify GPU IDs to use (e.g., "0,1,2,3"). If not specified, use all available GPUs')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from the latest checkpoint')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Directory containing checkpoints. If not specified, uses the default save directory')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Specific checkpoint file to resume from. Overrides automatic latest checkpoint detection')
    args = parser.parse_args()
    return args
def setup_gpu_environment(args):
    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
        gpu_count = len(args.gpus.split(','))
    else:
        gpu_count = torch.cuda.device_count()
    return gpu_count
def main():
    #Main function to run the neural rendering model with multi-GPU training.
    args = parse_arguments()
    gpu_count = setup_gpu_environment(args)
    print("=== Starting Neural Rendering Model with Multi-GPU Training ===")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print(f"Using {gpu_count} GPUs for training")
    print("================================================================")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Multi-GPU training requires CUDA.")
    if gpu_count < 1:
        raise RuntimeError("No GPUs available for training.")
    if gpu_count == 1:
        print("Using single GPU training...")
        run_dtu.run_model(**vars(args))
        return
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    print(f"Starting multi-GPU training on {gpu_count} GPUs...")
    mp.spawn(
        run_dtu.run_model_ddp,
        args=(gpu_count, args.train_folder, args.home_dir, args.dataset,
              args.baseres, args.scale, args.batch_size, args.resume,
              args.checkpoint_dir, args.checkpoint_path),
        nprocs=gpu_count,
        join=True
    )
    print("Training completed successfully!")
if __name__ == "__main__":
    main()

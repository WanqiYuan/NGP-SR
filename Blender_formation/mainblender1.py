#!/usr/bin/env python
import argparse
import os
import torch
import run_blender  

def parse_arguments():
    #Parse command line arguments with default values from the parameter file.
    parser = argparse.ArgumentParser(description='Neural Rendering Model Training and Evaluation')
    
    # Dataset parameters
    parser.add_argument('--train_folder', type=str, default="lego", help='Training folder name')
    parser.add_argument('--home_dir', type=str, default="/project/nianyil/ailab/ICCV2025/nerf_synthetic", 
                        help='Home directory path')
    parser.add_argument('--dataset', type=str, default="Blender", help='Dataset name')
    parser.add_argument('--baseres', type=int, default=100, help='Base resolution')
    parser.add_argument('--scale', type=int, default=2, help='Scale factor')
    parser.add_argument('--batch_size', type=int, default=8, help='Scale factor')
    parser.add_argument('--num_mf_layers', type=int, default=1, help='number of mf')
    
    args = parser.parse_args()

    
    return args

def main():
    #Main function to run the neural rendering model.
    # Parse command-line arguments
    args = parse_arguments()
    
    print("=== Starting Neural Rendering Model with parameters ===")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("=====================================================")
    
    # Call the main function from the model code (File B)
    # Pass all the arguments as a dictionary
    run_blender.run_model(**vars(args))

if __name__ == "__main__":
    main()
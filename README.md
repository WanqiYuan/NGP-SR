# NGP-SR

Official PyTorch implementation of **NGP-SR**. This repository provides training code for DTU and Blender/LLFF datasets.

[[Project page]](https://wanqiyuan.github.io/NGPSR-project-page/)


---

## 1. Environment Setup

Tested with **Python 3.x**.

**Core dependencies**:

- PyTorch 2.6.0
- CUDA 11.8
- matplotlib
- tqdm
- torchmetrics

Example setup (conda):

```bash
conda create -n ngpsr python=3.10
conda activate ngpsr

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib tqdm torchmetrics
```

Adjust the Python and CUDA versions according to your local environment if needed.

---

## 2. Data Structure

All datasets (DTU, Blender, LLFF) are converted into a unified Blender-style format.

An example directory layout is as follows (you can adapt this to your own paths):

```text
NGP-SR/
├── data/
│   ├── dtu/
│   │   ├── train/
│   │   │   ├── scan10/
│   │   │   │   ├── 0/                  # different light conditions
│   │   │   │   │   ├── images/
│   │   │   │   │   └── transforms.json # camera information for this light
│   │   │   │   ├── 1/
│   │   │   │   ├── 2/
│   │   │   │   ├── 3/
│   │   │   │   ├── 4/
│   │   │   │   ├── 5/
│   │   │   │   ├── 6/
│   │   │   │   └── transforms.json      # subset of selected views over all lights
│   │   │   └── ...
│   │   └── test/
│   │       ├── scan1/
│   │       ├── scan20/
│   │       └── ...
│   ├── blender/
│   │   ├── chair/
│   │   │   ├── train/               
│   │   │   ├── val/                 
│   │   │   ├── test/                
│   │   │   └── transforms_*.json    
│   │   └── ...                      # other Blender scenes
│   └── llff/
│       ├── fern/
│       │   ├── train/             
│       │   ├── val/     
│       │   ├── test/
│       │   └── transforms_*.json
│       └── ...                      # other LLFF scenes
└── ...
```

You can freely change the directory layout and adapt the dataloading logic to match your own preprocessing pipeline.

---

## 3. Training

### 3.1 Blender / LLFF Datasets

Example command for training on a Blender scene (e.g., `chair`):

```bash
python mainblender.py \
		--train_folder chair \
		--baseres 100 \
		--scale 2 \
		--batch_size 20 \
		--num_mf_layers 1
```

Key arguments (Blender/LLFF):

- `--train_folder`: scene name under `data/blender` or `data/llff`.
- `--baseres`: base spatial resolution for the low-resolution inputs.
- `--scale`: super-resolution upscale factor.
- `--batch_size`: number of rays or patches per batch (depends on implementation).
- `--num_mf_layers`: number of shared hash table layers.

### 3.2 DTU Dataset

Single-GPU training example:

```bash
python maindtus.py \
		--baseres 128 \
		--scale 4 \
		--batch_size 3 \
		--resume \
		--gpus "0"
```

Multi-GPU training (DTU is large, so we support multi-card training):

```bash
python maindtus.py \
		--baseres 128 \
		--scale 4 \
		--batch_size 3 \
		--resume \
		--gpus "0,1,2,3"
```

where `--gpus` specifies the visible GPU indices. Adjust `--batch_size` and `--baseres` according to your GPU memory.

---

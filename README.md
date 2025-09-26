# WheatVision

## Quickstart

```bash
# Create a new conda environment with Python 3.11+
conda create -n wheatvision python=3.11 -y
conda activate wheatvision

# Install requirements
pip install -r requirements.txt

# Launch Gradio app
python -m wheatvision.app
```

## Optional: Enable SAM2 Segmentation (Add-on)

SAM2 is treated as an **optional add-on**. It is not vendored into this repo and won’t be committed. You install it next to the project and the Segmentation tab will light up automatically if it’s available.


### 1) Install PyTorch/torchvision for your CUDA/OS

Follow https://pytorch.org/

```bash
# Example (adjust to your CUDA/OS):
pip install --pre torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

### 2) Clone & install SAM2 (editable) next to this repo

```bash
mkdir -p external
git clone https://github.com/facebookresearch/sam2.git external/sam2_repo
pip install -e external/sam2_rep
```

### 3) Download a checkpoint

Either use their helper script from the `sam2/checkpoints` folder or download one directly, then place it in a convenient folder, e.g.:

```bash
bash external/sam2_repo/checkpoints/download_ckpts.sh
```

### 4) Configuring SAM2

Configuration lives in `.env` at the project root.

```bash
WHEATVISION_SAM2_REPO=external/sam2_repo
WHEATVISION_SAM2_CFG=external/sam2_repo/sam2/configs/sam2.1/sam2.1_hiera_s.yaml
WHEATVISION_SAM2_CKPT=external/sam2_repo/checkpoints/sam2.1_hiera_small.pt
WHEATVISION_SAM2_DEVICE=cuda
```

### 5) Run the app

```
python -m wheatvision.app
```

If SAM2 is importable and the env vars are valid, the Segmentation tab will enable point-prompt inference. If SAM2 isn’t installed, you’ll see a gentle notice inside the tab and can still try Auto Mask (Otsu) as a baseline.

### Troubleshooting

- “Failed to build the SAM 2 CUDA extension” during install: you can usually ignore it (some post-processing features are limited, core results are fine).

- Torch/CUDA mismatch: ensure your installed torch matches your system CUDA or use CPU (slower).

- Windows: prefer WSL Ubuntu for compatibility.

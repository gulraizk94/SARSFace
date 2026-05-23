# SARSFace — Facial Detail Processing Module

Part of the SARS 3D Reconstruction Pipeline. This module handles high-level facial feature extraction including age, gender, landmarks, and detailed 3D face shape processing.

---

## Overview

- 3D facial detail extraction and manipulation
- Bilinear face model integration (FaceScape)
- Landmark-guided face alignment
- Detail map estimation and rendering
- CPU and GPU execution supported

---

## System Requirements

| Requirement | Details |
|---|---|
| OS | Linux (tested on CentOS 7 HPC, CUDA 11) |
| GPU | NVIDIA GPU (optional — CPU mode supported) |
| Python | 3.7 (see note below) |
| Conda | Anaconda or Miniconda |
| Extra tools | Blender 3.2, FFmpeg |

> ⚠️ **Python version note:** The original codebase targets Python 3.7. If you are running this inside a shared HPC environment with Python 3.10 (e.g. the `nanovlm` conda environment), see the [Python 3.10 Compatibility](#-python-310-compatibility-fixes) section below before installing.

---

## Environment Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/SARSFace.git
cd SARSFace
```

### 2. Create and activate conda environment

```bash
conda create -n semm python=3.7 -y
conda activate semm
```

### 3. Install dependencies

**Option A — via requirements file (recommended):**

```bash
pip install -r requirements.txt
```

**Option B — manual install:**

```bash
# PyTorch
pip install torch==1.9.0 torchvision==0.10.0

# PyTorch3D
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.6.2"

# Face detection and alignment
pip install ibug-face-detection ibug-face-alignment

# Other dependencies
pip install numpy==1.23.5 scipy==1.10.1 opencv-python==4.8.1.78
pip install trimesh scikit-learn scikit-image
```

---

##  Download Pretrained Models & Data

Place all downloaded files as described below **before** running any processing scripts.

### Model Checkpoints → `./checkpoints/`

| File | Source |
|---|---|
| `SEMM/` (unzip into checkpoints) | [Google Drive](https://drive.google.com/file/d/16g8zcvQXts9SuU5tgstHpWMgQ49vBmeY/view?usp=sharing) |
| `model_mse.pth` | [Google Drive](https://drive.google.com/file/d/1lc3GsP8XfIMDJfvamMmou2sOTG0ID02p/view?usp=sharing) |
| `dpmap_single_net_G.pth` | [Google Drive](https://drive.google.com/file/d/18j8bnj5IHP0u2jNuIrWh7dvQkfagBxsM/view?usp=sharing) |

```
./checkpoints/
├── SEMM/
├── model_mse.pth
└── dpmap_single_net_G.pth
```

### FaceScape Bilinear Model → `./predef/`

1. Download `facescape_bilinear_model_v1_3.zip` from [facescape.nju.edu.cn](https://facescape.nju.edu.cn/Page_Download/)
2. Extract and copy `core_847_50_52.npy` into `./predef/`

```
./predef/
└── core_847_50_52.npy
```

---

##  Install Blender & FFmpeg

Install **Blender 3.2** and **FFmpeg** and ensure both are accessible from the shell:

```bash
blender --version
ffmpeg -version
```

If they are not on your PATH, set the paths manually in:

```
./experiments/both_cond_launcher.py
```

```python
blender_path = "/path/to/blender"
ffmpeg_path  = "/path/to/ffmpeg"
```

### Install scipy inside Blender's Python

```bash
# Download pip for Blender's bundled Python
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
${blender_path}/3.2/python/bin/python3.10 get-pip.py

# Install scipy
${blender_path}/3.2/python/bin/python3.10 -m pip install scipy
```

---

## Usage

### Data Preprocessing

Run this before any editing tasks:

```bash
# GPU mode
python -m face_process.detail_process --input ./demo/input --output ./demo/output

# CPU mode (HPC without GPU allocation)
python -m face_process.detail_process --input ./demo/input --output ./demo/output --gpu_ids -1
```

---

## 🐛 Python 3.10 Compatibility Fixes

If you are using Python 3.10 (e.g. inside the `nanovlm` HPC environment), the following version conflicts must be resolved manually.

### Fix 1 — numpy + scipy conflict

The root cause is that newer `scipy` (≥1.11) requires `numpy ≥ 1.23.5` but also uses internal APIs that break on mismatched installs. Install both together:

```bash
pip install --upgrade --force-reinstall numpy==1.23.5 scipy==1.10.1
```

> Do **not** install them separately — pip may silently install a different numpy version than requested.

### Fix 2 — OpenCV version

```bash
pip install --upgrade --force-reinstall opencv-python==4.8.1.78
```

### Fix 3 — scikit-learn (sklearn.mixture.gaussian_mixture error)

Old scikit-learn versions (≤0.24) cannot be built from source on Python 3.10 due to missing `pkg_resources`. Install via conda instead:

```bash
conda install -c conda-forge scikit-learn=1.0.2 -y
```

> Do **not** try `pip install scikit-learn==0.18`, `0.19`, or `0.24` on Python 3.10 — these will all fail with `ModuleNotFoundError: No module named 'pkg_resources'`.

### Fix 4 — trimesh / scipy ufunc error

If you see `ValueError: All ufuncs must have type numpy.ufunc` from `scipy.special`, it means scipy and numpy versions are mismatched. Re-run Fix 1 above, then verify:

```bash
python -c "import numpy, scipy; print(numpy.__version__, scipy.__version__)"
# Expected: 1.23.5  1.10.1
```

### Verified working combination for Python 3.10

```
numpy==1.23.5
scipy==1.10.1
opencv-python==4.8.1.78
scikit-learn==1.0.2   (installed via conda)
```

---

## Environment Verification

```bash
python -c "import torch, cv2, trimesh, smplx, sklearn; print('All modules OK')"
blender --version
ffmpeg -version
```

---


---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

# DUSt3R Installation Summary

**Installation Date:** 2026-01-28
**Status:** ✅ **Successfully Installed and Verified**

## Environment Details

- **Python Version:** 3.11.14
- **Conda Environment:** `dust3r` (located at `/media/bob/data/miniconda3/envs/dust3r`)
- **PyTorch Version:** 2.5.1+cu121
- **CUDA Version:** 13.1 (driver: 590.48.01)
- **GPU:** NVIDIA GeForce RTX 3060 (12GB)

## Installed Components

### Core Dependencies ✅
- torch==2.5.1+cu121
- torchvision==0.20.1+cu121
- roma==1.5.4
- gradio==6.4.0
- matplotlib==3.10.8
- tqdm==4.67.1
- opencv-python==4.13.0.90
- scipy==1.17.0
- einops==0.8.2
- trimesh==4.11.1
- tensorboard==2.20.0
- pyglet==1.5.31
- huggingface-hub==1.3.4

### Optional Dependencies ✅
- pillow-heif==1.2.0 (HEIC image support)
- pyrender==0.1.45 (depth rendering)
- kapture==1.1.10 (visloc data loading)
- kapture-localization==1.1.10
- numpy-quaternion==2024.0.13
- pycolmap==3.13.0 (PnP pose estimation)
- poselib==2.0.5 (PnP pose estimation)

### CUDA Kernels ✅
- **RoPE CUDA kernels:** Successfully compiled at `croco/models/curope/curope.cpython-311-x86_64-linux-gnu.so`
- **Compilation:** Used CUDA 12.1 with g++ and nvcc
- **Architecture support:** sm_50, sm_60, sm_70, sm_75, sm_80, sm_86, sm_90

### Model Checkpoints ✅
- **DUSt3R_ViTLarge_BaseDecoder_512_dpt:** Auto-downloaded from HuggingFace Hub
- **Model ID:** naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt
- **Storage:** HuggingFace cache directory

## Verification Tests

### 1. Basic Imports ✅
```python
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.inference import inference
from dust3r.utils.image import load_images
# Result: All imports successful
```

### 2. CUDA Availability ✅
```python
import torch
print(torch.cuda.is_available())  # True
print(torch.version.cuda)  # 12.1
```

### 3. CUDA RoPE Kernels ✅
```python
from croco.models.curope import curope
print(dir(curope))  # ['rope_2d', ...]
# Result: CUDA module loaded successfully
```

### 4. Model Loading ✅
```python
model = AsymmetricCroCo3DStereo.from_pretrained('naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt')
model.to('cuda')
# Result: Model loaded on cuda:0
```

### 5. Full Inference Pipeline ✅
```python
# Test with croco/assets/Chateau1.png and Chateau2.png
images = load_images(['Chateau1.png', 'Chateau2.png'], size=512)
pairs = make_pairs(images, scene_graph='complete', symmetrize=True)
output = inference(pairs, model, 'cuda', batch_size=1)
# Result: Inference successful with 2 views
```

## Usage

### Activate Environment
```bash
# Quick activation
source /home/bob/Development/Ego3DT/dust3r/activate_dust3r.sh

# Or manually
conda activate dust3r
cd /home/bob/Development/Ego3DT/dust3r
```

### Run Interactive Demo
```bash
python demo.py --model_name DUSt3R_ViTLarge_BaseDecoder_512_dpt --device cuda
```

This launches a Gradio web interface at http://127.0.0.1:7860

### Run on LAN
```bash
python demo.py --model_name DUSt3R_ViTLarge_BaseDecoder_512_dpt --device cuda --local_network
```

### Basic Inference Script
```python
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
import torch

device = 'cuda'
model = AsymmetricCroCo3DStereo.from_pretrained(
    "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
).to(device)

images = load_images(['img1.jpg', 'img2.jpg'], size=512)
pairs = make_pairs(images, scene_graph='complete', symmetrize=True)
output = inference(pairs, model, device, batch_size=1)

# Global alignment
scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
loss = scene.compute_global_alignment(init="mst", niter=300, schedule="cosine", lr=0.01)

# Get results
pts3d = scene.get_pts3d()
focals = scene.get_focals()
poses = scene.get_im_poses()
confidence_masks = scene.get_masks()
```

## Known Issues

### Minor Warnings (Non-Breaking)
1. **FutureWarning:** `torch.cuda.amp.autocast(args...)` is deprecated
   - Impact: None (still works correctly)
   - Will be fixed in future PyTorch versions

2. **Ninja Build Warning:** Ninja not found, using distutils backend
   - Impact: Slightly slower compilation (one-time)
   - Optional optimization

3. **GCC Version Warning:** No version bounds for CUDA 12.1
   - Impact: None (compilation succeeded)
   - GCC compatibility verified

## Performance

- **Inference Speed:** ~0.5-2 iterations/second (GPU-accelerated)
- **Memory Usage:** ~2-4GB GPU memory for 512×512 images
- **Model Size:** ~2GB

## Integration with Ego3DT

DUSt3R is ready for integration into the Ego3DT pipeline:

1. **Stage 1-2:** GLEE detection + SAM segmentation (already implemented)
2. **Stage 3:** DUSt3R 3D reconstruction ← **Ready to implement**
3. **Stage 4:** Cross-window tracking ← Next step

### Next Steps for Integration

1. Create wrapper module: `src/reconstruction.py`
2. Connect GLEE+SAM outputs to DUSt3R inputs
3. Process video frames in sliding windows (W=30, overlap=10)
4. Extract object-specific 3D points using SAM masks
5. Implement cross-window tracking with Hungarian algorithm

## Troubleshooting

### CUDA Not Available
```bash
# Check CUDA installation
nvidia-smi
# Reinstall PyTorch with correct CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Import Errors
```bash
# Ensure correct environment is activated
conda activate dust3r
# Verify Python version
python --version  # Should be 3.11.x
```

### Out of Memory
```python
# Reduce batch size
output = inference(pairs, model, device, batch_size=1)
# Or reduce image size
images = load_images(['img1.jpg', 'img2.jpg'], size=224)
```

## Environment Backup

To recreate this environment:
```bash
# Export environment
conda env export > environment.yml

# Recreate from export
conda env create -f environment.yml
```

## Disk Space

- **Environment size:** ~6-7 GB
- **Model checkpoints:** ~2 GB
- **Total:** ~8-9 GB

## References

- DUSt3R Paper: https://arxiv.org/abs/2312.14132
- GitHub: https://github.com/naver/dust3r
- HuggingFace: https://huggingface.co/naver
- Project Documentation: `/home/bob/Development/Ego3DT/dust3r/CLAUDE.md`

---

**Installation completed by:** Claude Code CLI
**Verification status:** All tests passed ✅

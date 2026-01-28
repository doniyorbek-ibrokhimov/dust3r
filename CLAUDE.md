# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

DUSt3R (Geometric 3D Vision Made Easy) is a research codebase for 3D reconstruction from images. It uses a dual-encoder architecture (based on CroCo) to predict 3D point clouds directly from image pairs, with optional global alignment for multi-view reconstruction.

**License**: CC BY-NC-SA 4.0 (non-commercial use only)

**Related projects**: MASt3R, Pow3R, MUSt3R (see README for links)

## Development Commands

### Setup

```bash
# Python 3.11 required
pip install cmake==3.14.0
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -r requirements_optional.txt  # Optional: HEIC, pyrender, visloc

# Optional: Compile CUDA kernels for RoPE (faster runtime)
cd croco/models/curope/
python setup.py build_ext --inplace
cd ../../../
```

### Running

```bash
# Interactive demo (Gradio web interface)
python3 demo.py --model_name DUSt3R_ViTLarge_BaseDecoder_512_dpt

# Use --weights for local checkpoint
# Use --image_size for resolution (224 or 512)
# Use --local_network to expose on LAN
# Use --device to specify device

# Visual localization experiments
python3 visloc.py --model_name DUSt3R_ViTLarge_BaseDecoder_512_dpt \
    --dataset "VislocAachenDayNight('/path/to/dataset', ...)" \
    --pnp_mode poselib --output_dir /path/to/output
```

### Training

Training is done in 3 stages: 224px resolution with linear head → 512px with linear head → 512px with DPT head.

```bash
# Download pretrained CroCo checkpoint
wget https://download.europe.naverlabs.com/ComputerVision/CroCo/CroCo_V2_ViTLarge_BaseDecoder.pth -P checkpoints/

# Stage 1: 224 resolution
torchrun --nproc_per_node=4 train.py \
    --train_dataset "..." \
    --test_dataset "..." \
    --model "AsymmetricCroCo3DStereo(..., img_size=(224, 224), head_type='linear', ...)" \
    --pretrained "checkpoints/CroCo_V2_ViTLarge_BaseDecoder.pth" \
    --lr 0.0001 --epochs 10 --batch_size 16 \
    --output_dir "checkpoints/dust3r_224"

# Stage 2: 512 resolution (load from stage 1)
# Stage 3: 512 with DPT head (load from stage 2)
# See README "Training" section for full commands
```

### Docker

```bash
cd docker
bash run.sh --with-cuda --model_name="DUSt3R_ViTLarge_BaseDecoder_512_dpt"
# Or without CUDA: bash run.sh --model_name="..."
```

## Architecture

### Model Structure

**AsymmetricCroCo3DStereo** (`dust3r/model.py`): Main model class
- Two siamese ViT encoders (shared weights) process image pairs
- Two separate decoders output 3D points for each image
- Both outputs are in view1's reference frame (hence "asymmetric")
- Inherits from CroCo (`croco/models/croco.py` submodule)

**Encoder**: ViT-Large (1024 dim, 24 layers, 16 heads) with RoPE positional embeddings
- Custom patch embedding: `PatchEmbedDust3R` supports multiple aspect ratios (dust3r/patch_embed.py)
- No CLS token; uses RoPE instead of learned positional embeddings

**Decoder**: ViT-Base (768 dim, 12 layers, 12 heads)
- `dec_blocks` for view1, `dec_blocks2` for view2
- Cross-attention between views during decoding

**Heads** (`dust3r/heads/`):
- `LinearPts3d`: Simple linear projection to 3D + confidence
- `DPT` (Dense Prediction Transformer): Hierarchical decoder for higher quality

### Inference Pipeline

1. **Load images**: `dust3r.utils.image.load_images()` - handles various formats
2. **Make pairs**: `dust3r.image_pairs.make_pairs()` - scene graphs: 'complete', 'swin', 'logwin'
3. **Inference**: `dust3r.inference.inference()` - batched forward passes
   - Returns raw predictions: `view1`, `view2`, `pred1`, `pred2`
   - `pred1['pts3d']`: 3D points for view1 in view1's frame
   - `pred2['pts3d_in_other_view']`: 3D points for view2 in view1's frame
   - Both include confidence: `pred1['conf']`, `pred2['conf']`
4. **Global alignment** (`dust3r/cloud_opt/`):
   - `PairViewer`: No optimization, just visualizes raw output (1-2 images)
   - `PointCloudOptimizer`: Full global optimization (multi-view)
   - `ModularPointCloudOptimizer`: Alternative optimizer
   - `compute_global_alignment()`: Optimizes camera poses and point cloud jointly
     - Initialization modes: 'mst' (minimum spanning tree), 'known_poses'
     - Optimization schedules: 'cosine', 'linear'

### Training

**Loss**: `ConfLoss(Regr3D(L21))` (dust3r/losses.py)
- L2-L1 loss on 3D point predictions
- Confidence weighting
- Symmetric: computed on both (view1→view2) and (view2→view1)

**Training script**: `train.py` → `dust3r/training.py`
- Uses PyTorch DDP for multi-GPU
- Dataset string eval syntax: `"1000 @ DatasetName(param=value, ...)"`
- Supports multiple aspect ratios during training

### Datasets

**Base class**: `BaseStereoViewDataset` (dust3r/datasets/base/base_stereo_view_dataset.py)
- All datasets inherit and implement `_get_views(idx, resolution, rng)`
- Returns list of 2 view dicts with: `img`, `depthmap`, `camera_pose`, `camera_intrinsics`, `instance`

**Supported datasets** (dust3r/datasets/):
- CO3Dv2, ARKitScenes, ScanNet++, BlendedMVS, WayMo, Habitat-Sim, MegaDepth, StaticThings3D, WildRGB-D
- Each has preprocessing script in `datasets_preprocess/`
- Download pair lists from Naver Labs (see README Training section)

### CroCo Submodule

**Location**: `croco/` (git submodule)
- Contains base ViT architecture, blocks, positional embeddings
- RoPE implementation: `croco/models/curope/` (with optional CUDA kernels)
- Path injection: `dust3r/utils/path_to_croco.py` adds croco to sys.path

## Key Files

- `demo.py`: Gradio web interface entry point
- `train.py`: Training entry point
- `visloc.py`: Visual localization experiments entry point
- `dust3r/model.py`: Main model definition
- `dust3r/inference.py`: Inference utilities
- `dust3r/image_pairs.py`: Scene graph generation
- `dust3r/cloud_opt/`: Global alignment optimizers
- `dust3r/viz.py`: 3D visualization with plotly
- `dust3r_visloc/`: Visual localization datasets and evaluation

## Development Notes

### Model Checkpoints

Models auto-download from HuggingFace Hub via `AsymmetricCroCo3DStereo.from_pretrained()`.

For local checkpoints, use:
```python
model = AsymmetricCroCo3DStereo.from_pretrained('/path/to/checkpoint.pth')
# or via load_model() helper
```

Checkpoint format: `{'model': state_dict, 'args': model_args_string, ...}`

### Resolution Handling

Models are trained on specific resolutions. The 512 models support multiple aspect ratios:
- (512, 384), (512, 336), (512, 288), (512, 256), (512, 160)
- Uses `ManyAR_PatchEmbed` (aliased as `PatchEmbedDust3R`)

### Coordinate Frames

- Each image has its own camera frame
- `pts3d`: 3D points in the image's own frame
- `pts3d_in_other_view`: 3D points transformed to the other image's frame
- Global alignment optimizes to find a shared global frame

### Symmetrization

Training and inference can use symmetrized pairs: both (img1, img2) and (img2, img1).
- Enables optimization: forward pass computes only half the pairs
- Check with `is_symmetrized(view1, view2)`

### Confidence

Confidence values go through transformation (log-space) for optimization.
- `conf_mode=('exp', 1, inf)`: exponential confidence in [1, inf]
- Used as weights in loss and for filtering during visualization

## Visual Localization (visloc)

Separate module for camera pose estimation from query images:
- Datasets: Aachen-Day-Night, InLoc, Cambridge Landmarks, 7-Scenes
- Pipeline: DUSt3R inference → PnP solver (poselib) → pose refinement
- See `dust3r_visloc/README.md` for dataset preparation

## Common Patterns

### Basic Inference

```python
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

model = AsymmetricCroCo3DStereo.from_pretrained("naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt").to(device)
images = load_images(['img1.jpg', 'img2.jpg'], size=512)
pairs = make_pairs(images, scene_graph='complete', symmetrize=True)
output = inference(pairs, model, device, batch_size=1)

scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
loss = scene.compute_global_alignment(init="mst", niter=300, schedule="cosine", lr=0.01)

# Get results
imgs = scene.imgs
focals = scene.get_focals()
poses = scene.get_im_poses()
pts3d = scene.get_pts3d()
confidence_masks = scene.get_masks()
```

### Adding New Datasets

1. Create `dust3r/datasets/my_dataset.py` inheriting from `BaseStereoViewDataset`
2. Implement `_get_views(self, idx, resolution, rng)` returning list of 2 view dicts
3. Optionally create preprocessing script in `datasets_preprocess/`
4. Import in `dust3r/datasets/__init__.py`
5. Use in training with string eval syntax: `"MyDataset(ROOT='/path', ...)"`

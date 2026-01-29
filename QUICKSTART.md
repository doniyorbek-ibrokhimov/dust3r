# DUSt3R Quick Start Guide

## Activation

```bash
# Option 1: Use activation script
source /home/bob/Development/Ego3DT/dust3r/activate_dust3r.sh

# Option 2: Manual activation
conda activate dust3r
cd /home/bob/Development/Ego3DT/dust3r
```

## Common Commands

### Interactive Demo
```bash
# Local only (default: http://127.0.0.1:7860)
python demo.py --model_name DUSt3R_ViTLarge_BaseDecoder_512_dpt --device cuda

# Expose on LAN
python demo.py --model_name DUSt3R_ViTLarge_BaseDecoder_512_dpt --device cuda --local_network

# Use smaller model/resolution
python demo.py --model_name DUSt3R_ViTLarge_BaseDecoder_224_linear --image_size 224 --device cuda
```

### Python Inference

#### Basic Two-Image Reconstruction
```python
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs

# Load model
model = AsymmetricCroCo3DStereo.from_pretrained(
    "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
).to('cuda')

# Load images
images = load_images(['img1.jpg', 'img2.jpg'], size=512)

# Create pairs
pairs = make_pairs(images, scene_graph='complete', symmetrize=True)

# Run inference
output = inference(pairs, model, 'cuda', batch_size=1)
```

#### Multi-View Reconstruction with Global Alignment
```python
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

# Run inference first (as above)
# ...

# Global alignment
scene = global_aligner(
    output,
    device='cuda',
    mode=GlobalAlignerMode.PointCloudOptimizer
)

loss = scene.compute_global_alignment(
    init="mst",      # initialization: 'mst' or 'known_poses'
    niter=300,       # optimization iterations
    schedule="cosine",  # learning rate schedule
    lr=0.01          # learning rate
)

# Extract results
pts3d = scene.get_pts3d()              # 3D point clouds
focals = scene.get_focals()            # camera focal lengths
poses = scene.get_im_poses()           # camera poses (4×4 matrices)
confidence_masks = scene.get_masks()   # confidence masks
imgs = scene.imgs                      # images
```

### Ego3DT Integration Example

```python
import numpy as np
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs

# Load DUSt3R model
dust3r_model = AsymmetricCroCo3DStereo.from_pretrained(
    "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
).to('cuda')

# Window of frames from Ego3DT
window_frames = [...]  # List of 30 frames
window_masks = [...]   # SAM masks from GLEE+SAM pipeline

# Convert frames to format for DUSt3R
images = []
for frame in window_frames:
    # Prepare image dict
    images.append({
        'img': frame,
        'true_shape': frame.shape[:2],
        'idx': len(images),
        'instance': str(len(images))
    })

# Create pairs (e.g., sliding window graph)
pairs = make_pairs(images, scene_graph='swin', prefilter=None, symmetrize=True)

# Run DUSt3R
output = inference(pairs, dust3r_model, 'cuda', batch_size=1)

# Extract 3D points
pts3d_per_image = []
for idx, pred in enumerate([output['pred1'], output['pred2']]):
    pts3d = pred['pts3d']  # (H, W, 3)
    conf = pred['conf']    # (H, W)

    # Apply SAM mask to get object-specific 3D points
    mask = window_masks[idx]  # (H, W) boolean
    object_pts3d = pts3d[mask]

    # Calculate centroid
    centroid = np.mean(object_pts3d, axis=0)
    pts3d_per_image.append({
        'points': object_pts3d,
        'centroid': centroid,
        'num_points': len(object_pts3d)
    })
```

## Model Variants

| Model Name | Resolution | Decoder | Size | Speed | Quality |
|------------|-----------|---------|------|-------|---------|
| `DUSt3R_ViTLarge_BaseDecoder_224_linear` | 224×224 | Linear | ~2GB | Fast | Good |
| `DUSt3R_ViTLarge_BaseDecoder_512_linear` | 512×512 | Linear | ~2GB | Medium | Better |
| `DUSt3R_ViTLarge_BaseDecoder_512_dpt` | 512×512 | DPT | ~2GB | Slow | Best |

## Scene Graph Types

```python
# Complete graph: all pairs (best quality, slowest)
pairs = make_pairs(images, scene_graph='complete')

# Sliding window: temporal neighbors (good for video)
pairs = make_pairs(images, scene_graph='swin')

# Log-window: exponentially spaced pairs
pairs = make_pairs(images, scene_graph='logwin')
```

## Output Structure

```python
output = {
    'view1': {
        'idx': [0, 1, ...],          # image indices
        'instance': ['0', '1', ...], # image IDs
        'img': Tensor[N, 3, H, W],   # RGB images
    },
    'view2': { ... },  # same structure for second view
    'pred1': {
        'pts3d': Tensor[N, H, W, 3],           # 3D points in view1's frame
        'conf': Tensor[N, H, W],               # confidence scores
        'pts3d_in_other_view': Tensor[N, H, W, 3]  # optional
    },
    'pred2': {
        'pts3d_in_other_view': Tensor[N, H, W, 3],  # 3D points in view1's frame
        'conf': Tensor[N, H, W]
    }
}
```

## Visualization

### Using Built-in Visualizer
```python
from dust3r.viz import SceneViz

viz = SceneViz()
viz.add_pointcloud(pts3d, color, confidence)
viz.add_camera(pose, focal, color='red', size=0.1)
viz.show()
```

### Export to PLY
```python
import trimesh

# Create point cloud
points = pts3d.reshape(-1, 3)
colors = images[0].reshape(-1, 3)

cloud = trimesh.PointCloud(points, colors=colors)
cloud.export('output.ply')
```

## Performance Tips

1. **Reduce memory usage:**
   ```python
   # Smaller batch size
   output = inference(pairs, model, 'cuda', batch_size=1)

   # Lower resolution
   images = load_images(['img1.jpg', 'img2.jpg'], size=224)
   ```

2. **Speed up inference:**
   ```python
   # Use linear head model
   model = AsymmetricCroCo3DStereo.from_pretrained(
       "naver/DUSt3R_ViTLarge_BaseDecoder_512_linear"
   )

   # Disable symmetrization
   pairs = make_pairs(images, scene_graph='complete', symmetrize=False)
   ```

3. **Clear GPU memory:**
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

## Troubleshooting

### Out of Memory
- Reduce image size: `size=224` instead of `size=512`
- Use smaller batch size: `batch_size=1`
- Clear GPU cache: `torch.cuda.empty_cache()`

### Slow Inference
- Use linear decoder: `*_linear` model instead of `*_dpt`
- Reduce resolution to 224
- Use smaller scene graph: `'swin'` instead of `'complete'`

### Poor Quality Results
- Increase resolution to 512
- Use DPT decoder: `*_dpt` model
- Use complete scene graph
- Ensure good camera motion between images

## Next Steps

1. **Try the demo:**
   ```bash
   python demo.py --model_name DUSt3R_ViTLarge_BaseDecoder_512_dpt
   ```

2. **Test with your images:**
   ```python
   images = load_images(['your_img1.jpg', 'your_img2.jpg'], size=512)
   ```

3. **Integrate with Ego3DT:**
   - See `CLAUDE.md` for integration guide
   - Connect to GLEE+SAM pipeline
   - Implement sliding window processing

## Resources

- Full documentation: `CLAUDE.md`
- Installation details: `INSTALLATION_SUMMARY.md`
- Original README: `README.md`
- Paper: https://arxiv.org/abs/2312.14132

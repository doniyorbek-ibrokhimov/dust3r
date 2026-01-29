#!/usr/bin/env python
"""DUSt3R Installation Verification Script"""

import sys

def verify_installation():
    print('=== DUSt3R Installation Verification ===\n')

    all_passed = True

    # 1. Python version
    python_version = sys.version.split()[0]
    if python_version.startswith('3.11'):
        print(f'✓ Python: {python_version}')
    else:
        print(f'✗ Python: {python_version} (expected 3.11.x)')
        all_passed = False

    # 2. PyTorch and CUDA
    try:
        import torch
        print(f'✓ PyTorch: {torch.__version__}')

        if torch.cuda.is_available():
            print(f'✓ CUDA available: True')
            print(f'✓ CUDA version: {torch.version.cuda}')
            print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
            print(f'✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
        else:
            print('✗ CUDA not available')
            all_passed = False
    except Exception as e:
        print(f'✗ PyTorch import failed: {e}')
        all_passed = False

    # 3. Core DUSt3R imports
    try:
        from dust3r.model import AsymmetricCroCo3DStereo
        from dust3r.inference import inference
        from dust3r.utils.image import load_images
        from dust3r.image_pairs import make_pairs
        print('✓ DUSt3R core imports successful')
    except Exception as e:
        print(f'✗ DUSt3R imports failed: {e}')
        all_passed = False

    # 4. CUDA RoPE kernels
    try:
        from croco.models.curope import curope
        if 'rope_2d' in dir(curope):
            print('✓ CUDA RoPE kernels: compiled and available')
        else:
            print('⚠ CUDA RoPE kernels: module loaded but functions not found')
    except Exception as e:
        print(f'⚠ CUDA RoPE kernels: {e} (optional, will use PyTorch fallback)')

    # 5. HuggingFace Hub
    try:
        from huggingface_hub import hf_hub_download
        print('✓ HuggingFace Hub: available')
    except Exception as e:
        print(f'✗ HuggingFace Hub: {e}')
        all_passed = False

    # 6. Optional dependencies (don't affect overall status)
    print('\nOptional dependencies:')
    optional_imports = [
        ('pyrender', 'depth rendering'),
        ('poselib', 'PnP estimation'),
        ('pillow_heif', 'HEIC image support'),
        ('kapture', 'visloc data loading'),
    ]

    for module_name, description in optional_imports:
        try:
            __import__(module_name)
            print(f'  ✓ {module_name} ({description})')
        except ImportError:
            print(f'  - {module_name} ({description}) - not installed')

    # Final status
    print('\n' + '='*50)
    if all_passed:
        print('✓ Installation Status: SUCCESS')
        print('\nAll critical components are installed and working.')
        print('\nNext steps:')
        print('  1. Run the demo:')
        print('     python demo.py --model_name DUSt3R_ViTLarge_BaseDecoder_512_dpt')
        print('  2. See QUICKSTART.md for usage examples')
        print('  3. See CLAUDE.md for integration with Ego3DT')
        return 0
    else:
        print('✗ Installation Status: FAILED')
        print('\nSome critical components are missing.')
        print('Please check the error messages above.')
        return 1

if __name__ == '__main__':
    sys.exit(verify_installation())

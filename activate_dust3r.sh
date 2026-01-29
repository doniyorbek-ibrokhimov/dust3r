#!/bin/bash
# Activation script for DUSt3R environment

# Activate conda environment
source /media/bob/data/miniconda3/bin/activate dust3r

# Display environment info
echo "DUSt3R environment activated!"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Change to dust3r directory
cd /home/bob/Development/Ego3DT/dust3r

echo ""
echo "Usage examples:"
echo "  python demo.py --model_name DUSt3R_ViTLarge_BaseDecoder_512_dpt --device cuda"
echo "  python demo.py --local_network  # Expose on LAN"

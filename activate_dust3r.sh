#!/bin/bash
# Activation script for DUSt3R environment
# Usage: source dust3r/activate_dust3r.sh (from project root)

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Activate venv environment
source "$SCRIPT_DIR/.venv/bin/activate"

# Add dust3r directory to PYTHONPATH so imports work
export PYTHONPATH="$SCRIPT_DIR:$PROJECT_ROOT:$PYTHONPATH"

# Display environment info
echo "DUSt3R environment activated!"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'PyTorch not available')"
echo ""

# Verify dust3r module is available
if python -c "import dust3r.model" 2>/dev/null; then
    echo "✓ DUSt3R module found"
else
    echo "✗ DUSt3R module not found - check installation"
fi

echo ""
echo "Usage examples:"
echo "  python scripts/run_reconstruction.py --data_dir outputs/short_sample/ --output_dir outputs/reconstruction/"
echo "  cd dust3r && python demo.py --model_name DUSt3R_ViTLarge_BaseDecoder_512_dpt --device cuda"


create venv
```bash
deactivate

sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev

python3.11 -m venv .venv

source .venv/bin/activate

pip3 install cmake==3.14.3

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121


# Install required dependencies
pip install -r requirements.txt

# Optional: you can also install additional packages to:
# - add support for HEIC images
# - add pyrender, used to render depthmap in some datasets preprocessing
# - add required packages for visloc.py
pip install -r requirements_optional.txt


# DUST3R relies on RoPE positional embeddings for which you can compile some cuda kernels for faster runtime.
cd croco/models/curope/
python setup.py build_ext --inplace
cd ../../../

# Download checkpoint
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth -P checkpoints/

#Run demo
python demo.py --model_name DUSt3R_ViTLarge_BaseDecoder_512_dpt --device cuda


```
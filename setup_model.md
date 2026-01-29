
create venv
```bash
deactivate

sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev

python3.11 -m venv .venv

source .venv/bin/activate

pip3 install cmake==3.14.3

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121


# conda create -n dust3r python=3.11 cmake=3.14.0

# conda activate dust3r 

# conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia  # use the correct version of cuda for your system

pip install -r requirements.txt

# build croco
cd croco/models/curope/


```
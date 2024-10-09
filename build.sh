# python3.8 -m venv venv


./venv/bin/pip install --upgrade pip

./venv/bin/pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124

./venv/bin/pip install -r requirements.txt

./venv/bin/pip install basicsr facexlib realesrgan

./venv/bin/python setup.py develop

mkdir -p experiments/pretrained_models/

wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth -P experiments/pretrained_models/

sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' venv/lib/python3.8/site-packages/basicsr/data/degradations.py

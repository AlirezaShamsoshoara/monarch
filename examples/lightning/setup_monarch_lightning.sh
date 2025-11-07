#!/bin/bash
set -e  # Exit on error

# Prompt for API keys if not already set
if [ -z "$WANDB_API_KEY" ]; then
  read -p "Enter your WANDB_API_KEY: " WANDB_API_KEY
  export WANDB_API_KEY
fi

if [ -z "$HF_API_KEY" ]; then
  read -p "Enter your HF_API_KEY: " HF_API_KEY
  export HF_API_KEY
fi

# Install Python packages
pip install -U lightning_sdk

# Clone and install TorchTitan
git clone https://github.com/pytorch/torchtitan.git
cd torchtitan
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --force-reinstall
pip install .
pip install wandb
wandb login $WANDB_API_KEY

# Download Llama 3.1 tokenizer assets
python scripts/download_hf_assets.py --repo_id meta-llama/Llama-3.1-8B --assets tokenizer --hf_token=$HF_API_KEY

cd ~

# Clone and set up Monarch
git clone https://github.com/meta-pytorch/monarch.git
cd monarch

# Install Rust nightly toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env
rustup toolchain install nightly
rustup default nightly

# Install Ubuntu system dependencies
sudo apt install -y ninja-build libunwind-dev clang
sudo apt install -y net-tools
sudo apt install -y iputils-ping
sudo apt install -y lsof
sudo apt install -y netcat-traditional

# Set clang as default compiler
export CC=clang
export CXX=clang++

# Install Monarch Python dependencies
pip install -r build-requirements.txt
pip install -r python/tests/requirements.txt
USE_TENSOR_ENGINE=0 pip install --no-build-isolation .

# Cleanup steps
rm -rf $HOME/.cargo $HOME/.rustup
sed -i '/\.cargo\/env/d' $HOME/.bashrc
sed -i '/\.cargo\/env/d' $HOME/.zshenv
cd ~
rm -rf monarch

echo "Environment setup complete!"

# Verification steps (run in Python)
echo "Verifying installations..."
python -c "import torchtitan; print('TorchTitan is installed successfully')"
python -c "import monarch; print('Monarch is installed successfully')"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

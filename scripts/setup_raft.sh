#!/bin/bash

echo "Setting up RAFT workspace..."
mkdir -p raft_workspace
cd raft_workspace

if [ ! -d "RAFT" ]; then
    echo "Cloning RAFT repository..."
    git clone https://github.com/princeton-vl/RAFT.git
fi

if [ ! -d "models" ]; then
    echo "Downloading RAFT models..."
    mkdir models
    cd models
    wget https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip
    unzip models.zip
    rm models.zip
    cd ..
fi

echo "Creating and configuring Conda environment 'raft_env'..."
# Check if conda exists
if command -v conda &> /dev/null; then
    # Create environment if it doesn't exist
    conda create -n raft_env python=3.8 -y || echo "Environment raft_env might already exist."
    
    # Initialize conda for this script session
    eval "$(conda shell.bash hook)"
    
    echo "Activating raft_env..."
    conda activate raft_env
    
    echo "Installing PyTorch and dependencies..."
    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
    pip install opencv-python matplotlib scipy tensorboard
    
    echo "Setup complete! You can now run: conda activate raft_env"
else
    echo "Conda not found. Please ensure Conda is installed and in your PATH."
fi

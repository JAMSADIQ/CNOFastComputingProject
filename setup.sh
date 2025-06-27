#!/bin/bash

# Name of the conda environment
ENV_NAME="cnn_env"
PYTHON_VERSION="3.10"

echo "Creating conda environment: $ENV_NAME with Python $PYTHON_VERSION"
conda create -y -n $ENV_NAME python=$PYTHON_VERSION

echo "Activating the environment"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

echo "Installing PyTorch and related libraries (CPU version)"
conda install -y pytorch torchvision torchaudio cpuonly -c pytorch

echo "Installing PyVista and visualization libraries"
conda install -y -c conda-forge pyvista pyvistaqt matplotlib scipy numpy

echo "Installing other commonly used CNN libraries"
conda install -y pandas scikit-learn jupyter notebook tqdm seaborn

echo "Installation complete. Environment '$ENV_NAME' is ready."


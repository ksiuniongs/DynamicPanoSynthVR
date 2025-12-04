#!/bin/bash
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
cd "$PROJECT_ROOT"

echo "Fetching code from github..."
# Use git sparse-checkout to download only the single_view_mpi directory
git clone --no-checkout --depth=1 https://github.com/google-research/google-research.git temp_repo
cd temp_repo
git sparse-checkout init
git sparse-checkout set single_view_mpi
git checkout
mv single_view_mpi ../
cd ..
rm -rf temp_repo

echo
echo "Fetching trained model weights..."
rm -f single_view_mpi_full_keras.tar.gz
rm -rf single_view_mpi_full
wget https://storage.googleapis.com/stereo-magnification-public-files/models/single_view_mpi_full_keras.tar.gz
tar -xzvf single_view_mpi_full_keras.tar.gz
rm single_view_mpi_full_keras.tar.gz

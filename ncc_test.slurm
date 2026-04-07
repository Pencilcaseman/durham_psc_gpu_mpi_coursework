#!/bin/bash
#SBATCH --job-name=mpi-cuda-run
#SBATCH --partition=ug-gpu-small
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:turing:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=slurm_output/slurm-%j.out

set -euxo pipefail

module load cuda/12.4
module load intel-oneapi/2024.1.0/mpi

mkdir -p slurm_build && pushd slurm_build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
popd

mpiexec -np 2 ./slurm_build/mpi_cuda_tests

#!/bin/bash
#SBATCH --job-name=mpi-cuda-run
#SBATCH --partition=ug-gpu-small
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:turing:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=slurm_output/slurm-%j.out

set -euxo pipefail

module load cuda
module load intel-oneapi/2024.1.0/mpi

mkdir -p slurm_build && pushd slurm_build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
popd

echo "========================================================================="
echo "AXPY"
echo "========================================================================="
srun -n 1 slurm_build/mpi_cuda_coursework \
    --mode axpy --N 50000000 --alpha 3.1415926 \
    --csv results.csv

echo "========================================================================="
echo "ADD"
echo "========================================================================="
srun -n 1 slurm_build/mpi_cuda_coursework \
    --mode add --N 50000000 \
    --csv results.csv

echo "========================================================================="
echo "COPY"
echo "========================================================================="
srun -n 1 slurm_build/mpi_cuda_coursework \
    --mode copy --N 50000000 \
    --csv results.csv

# srun -n 2 slurm_build/mpi_cuda_coursework \
#     --mode gemm --M 4096 --N 4096 --K 4096 \
#     --kernel tiled --csv results.csv

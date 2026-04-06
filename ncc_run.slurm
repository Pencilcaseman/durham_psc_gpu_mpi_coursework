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

export NUM_RANKS=1

module load cuda
module load intel-oneapi/2024.1.0/mpi

mkdir -p slurm_build && pushd slurm_build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
popd

echo "========================================================================="
echo "AXPY"
echo "========================================================================="
mpiexec -np $NUM_RANKS slurm_build/mpi_cuda_coursework \
    --mode axpy --N 50000000 --alpha 3.1415926 \
    --csv results.csv

echo "========================================================================="
echo "ADD"
echo "========================================================================="
mpiexec -np $NUM_RANKS slurm_build/mpi_cuda_coursework \
    --mode add --N 50000000 \
    --csv results.csv

echo "========================================================================="
echo "COPY"
echo "========================================================================="
mpiexec -np $NUM_RANKS slurm_build/mpi_cuda_coursework \
    --mode copy --N 50000000 \
    --csv results.csv

echo "========================================================================="
echo "REDUCE"
echo "========================================================================="
mpiexec -np $NUM_RANKS slurm_build/mpi_cuda_coursework \
    --mode reduce --N 50000000 \
    --csv results.csv

echo "========================================================================="
echo "GEMM (naive)"
echo "========================================================================="
mpiexec -np $NUM_RANKS slurm_build/mpi_cuda_coursework \
    --mode gemm --M 1024 --N 1024 --K 1024 \
    --kernel naive --csv results.csv

echo "========================================================================="
echo "GEMM (tiled)"
echo "========================================================================="
mpiexec -np $NUM_RANKS slurm_build/mpi_cuda_coursework \
    --mode gemm --M 1024 --N 1024 --K 1024 \
    --kernel tiled --csv results.csv

#!/bin/bash
#SBATCH --job-name=mpi-cuda-run
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=1          # single GPU shared by all ranks
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%j.out

module purge
module load cuda openmpi cmake

mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
cd ..

# All ranks share the single GPU (device 0 is picked automatically in code)
srun -n 2 build/mpi_cuda_coursework \
    --mode gemm --M 4096 --N 4096 --K 4096 \
    --kernel tiled --csv results.csv

#!/bin/bash
#SBATCH --job-name=mpi-cuda-scaling
#SBATCH --partition=ug-gpu-small
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:turing:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm_output/scaling-%j.out

set -euo pipefail

module purge
module load cuda/12.4
module load intel-oneapi/2024.1.0/mpi

mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
cd ..

CSV=scaling_results.csv

# Vector operations: vary size and ranks
for MODE in axpy add copy; do
    for N in 1000000 5000000 10000000 50000000 100000000; do
        for NP in 1 2 4; do
            echo "=== $MODE N=$N NP=$NP ==="
            mpiexec -np $NP build/mpi_cuda_coursework \
                --mode $MODE --N $N --csv $CSV
        done
    done
    echo ""
done

# Reduce: vary size and ranks
for N in 1000000 5000000 10000000 50000000 100000000; do
    for NP in 1 2 4; do
        echo "=== reduce N=$N NP=$NP ==="
        mpiexec -np $NP build/mpi_cuda_coursework \
            --mode reduce --N $N --csv $CSV
    done
    echo ""
done

# GEMM: vary size, kernel, and ranks
for KERNEL in naive tiled optimised; do
    for SIZE in 512 1024 2048 4096; do
        for NP in 1 2 4; do
            echo "=== gemm kernel=$KERNEL M=$SIZE N=$SIZE K=$SIZE NP=$NP ==="
            mpiexec -np $NP build/mpi_cuda_coursework \
                --mode gemm --M $SIZE --N $SIZE --K $SIZE \
                --kernel $KERNEL --csv $CSV
        done
        echo ""
    done

    echo ""
    echo ""
done

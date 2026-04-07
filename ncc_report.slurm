#!/bin/bash
#SBATCH --job-name=mpi-cuda-report
#SBATCH --partition=ug-gpu-small
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:turing:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm_output/report-%j.out

set -euo pipefail

module purge
module load cuda/12.4
module load intel-oneapi/2024.1.0/mpi

export TMPDIR=$PWD/tmp
mkdir -p $TMPDIR

mkdir -p slurm_build && pushd slurm_build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
popd

BIN=slurm_build/mpi_cuda_coursework
CSV=results.csv
rm -f $CSV

# ==========================================================================
# Part A1 -- Vector operations: bandwidth vs N, with MPI scaling
# ==========================================================================

for MODE in axpy add copy; do
    for N in 100000 500000 1000000 2000000 5000000 10000000 20000000 50000000; do
        for NP in 1 2 3 4; do
            echo "--- $MODE N=$N NP=$NP ---"
            mpiexec -np $NP $BIN --mode $MODE --N $N --csv $CSV
        done
    done
done

# ==========================================================================
# Part A2 -- Reduction: correctness + performance
# ==========================================================================

for N in 100000 500000 1000000 2000000 5000000 10000000 20000000 50000000; do
    for NP in 1 2 3 4; do
        echo "--- reduce N=$N NP=$NP ---"
        mpiexec -np $NP $BIN --mode reduce --N $N --csv $CSV
    done
done

# ==========================================================================
# Part B -- GEMM: naive vs tiled vs optimised, varying size and ranks
# ==========================================================================

# All kernels use the same sizes for consistent comparison.
# Naive at 16384 takes ~22s but is within budget.
for KERNEL in naive tiled optimised; do
    for SIZE in 256 512 768 1024 1536 2048 3072 4096 6144 8192 12288 16384; do
        for NP in 1 2 3 4; do
            echo "--- gemm $KERNEL M=$SIZE NP=$NP ---"
            mpiexec -np $NP $BIN \
                --mode gemm --M $SIZE --N $SIZE --K $SIZE \
                --kernel $KERNEL --csv $CSV
        done
    done
done

# ==========================================================================
# Part C2 -- Profiling: nsys trace of a multi-rank GEMM run
# ==========================================================================

mkdir -p nsys_profiles

for KERNEL in naive tiled optimised; do
    echo "--- profiling gemm $KERNEL NP=2 ---"
    mpiexec -np 2 nsys profile \
        --trace=cuda,mpi --mpi-impl=mpich --force-overwrite=true \
        --output=nsys_profiles/gemm_${KERNEL}_np2_rank%q{PMI_RANK} \
        $BIN --mode gemm --M 4096 --N 4096 --K 4096 \
        --kernel $KERNEL --csv $CSV
done

echo "Done. Results in $CSV, profiles in nsys_profiles/"

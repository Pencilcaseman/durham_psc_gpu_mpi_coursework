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
    for N in 1000000 5000000 10000000 50000000; do
        for NP in 1 2 3 4; do
            echo "--- $MODE N=$N NP=$NP ---"
            mpiexec -np $NP $BIN --mode $MODE --N $N --csv $CSV
        done
    done
done

# ==========================================================================
# Part A2 -- Reduction: correctness + performance
# ==========================================================================

for N in 1000000 5000000 10000000 50000000; do
    for NP in 1 2 3 4; do
        echo "--- reduce N=$N NP=$NP ---"
        mpiexec -np $NP $BIN --mode reduce --N $N --csv $CSV
    done
done

# ==========================================================================
# Part B -- GEMM: naive vs tiled vs optimised, varying size and ranks
# ==========================================================================

# Naive: cap at 8192 (16384 too slow for naive)
for SIZE in 512 1024 2048 4096 8192 16384; do
    for NP in 1 2 3 4; do
        echo "--- gemm naive M=$SIZE NP=$NP ---"
        mpiexec -np $NP $BIN \
            --mode gemm --M $SIZE --N $SIZE --K $SIZE \
            --kernel naive --csv $CSV
    done
done

# Tiled + optimised: can go larger
for KERNEL in tiled optimised; do
    for SIZE in 512 1024 2048 4096 8192 16384; do
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
        --trace=cuda,mpi --force-overwrite=true \
        --output=nsys_profiles/gemm_${KERNEL}_np2 \
        $BIN --mode gemm --M 4096 --N 4096 --K 4096 \
        --kernel $KERNEL --csv $CSV
done

echo "Done. Results in $CSV, profiles in nsys_profiles/"

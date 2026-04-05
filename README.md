# COMP3741 – MPI + CUDA Coursework Starter

## Project structure
```
.
├── CMakeLists.txt
├── README.md
├── include/          # All header files
│   ├── benchmarks.hpp
│   ├── check_cuda.hpp
│   ├── cli.hpp
│   ├── cpu_reference.hpp
│   ├── kernels.hpp
│   ├── mpi_distribution.hpp
│   ├── mpi_utils.hpp
│   └── timer.hpp
├── src/              # Implementation files
│   ├── benchmarks.cpp
│   ├── cpu_reference.cpp
│   ├── cuda_kernels.cu
│   ├── main.cpp
│   └── mpi_distribution.cpp
├── tests/
│   └── test_main.cpp
├── ncc_run.slurm     # Single run (2 ranks, 1 GPU)
└── ncc_scaling.slurm # Strong-scaling sweep (1/2/4 ranks, 1 GPU)
```

## Single-GPU assumption
All MPI ranks share **one GPU (device 0)**. Each rank allocates its own device
memory and launches its own CUDA kernels. No multi-GPU or peer-access code is
needed.

## Build (NCC cluster)
```bash
module purge
module load cuda openmpi cmake

mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
```

> **CUDA architecture**: edit `CUDA_ARCHITECTURES` in `CMakeLists.txt` if
> your GPU is not V100 (sm_70) or A100 (sm_80).  
> Run `nvidia-smi` to find the GPU model, then look up its compute capability.

## Run
```bash
# AXPY bandwidth test
mpirun -np 2 ./build/mpi_cuda_coursework --mode axpy --N 50000000 --csv results.csv

# Parallel reduction
mpirun -np 2 ./build/mpi_cuda_coursework --mode reduce --N 50000000 --csv results.csv

# GEMM – naive kernel
mpirun -np 2 ./build/mpi_cuda_coursework --mode gemm \
    --M 4096 --N 4096 --K 4096 --kernel naive --csv results.csv

# GEMM – tiled kernel
mpirun -np 2 ./build/mpi_cuda_coursework --mode gemm \
    --M 4096 --N 4096 --K 4096 --kernel tiled --csv results.csv
```

## Correctness tests
```bash
mpirun -np 2 ./build/mpi_cuda_tests
```

## SLURM (NCC)
```bash
sbatch ncc_run.slurm        # 2-rank run
sbatch ncc_scaling.slurm    # strong-scaling sweep
```

## Reproducibility
- Random data is seeded deterministically (`seed = base + rank`).
- Each CSV row is tagged with the MPI rank, mode, kernel, and problem size.
- Warm-up kernel launches precede all timed runs.

# COMP3741 -- MPI & CUDA Coursework

## Notes

- All MPI ranks share **GPU 0**
- The optimised GEMM converts to float16 for tensor cores and accumulates back
  into float32
- Warp size of 32 (NVIDIA GPUs)
- Compiled for **sm_75** (Turing)
- Random data seeded deterministically per rank (`seed = offset + rank`)
- GEMM validation is skipped for matrices exceeding 1M elements

## Build and Run (NCC)

```bash
sbatch ncc_run.slurm      # single-rank smoke test
sbatch ncc_report.slurm   # full benchmark suite + Nsight profiling
```

Or locally with [just](https://github.com/casey/just):

```bash
source setup.sh
just build release
```

## Modes

| Mode                      | Description                                  |
|---------------------------|----------------------------------------------|
| `axpy`                    | `y = a*x + y` bandwidth benchmark            |
| `add`                     | `z = x + y` bandwidth benchmark              |
| `copy`                    | `y = x` bandwidth benchmark                  |
| `reduce`                  | Parallel sum with warp-level tree reduction  |
| `gemm --kernel naive`     | One thread per output element, global memory |
| `gemm --kernel tiled`     | 32x32 shared-memory tiles                    |
| `gemm --kernel optimised` | F16 WMMA with double-buffered 128x128 tiles  |

## Correctness Tests

```bash
mpiexec -np 2 slurm_build/mpi_cuda_tests
```

## Output

- `results.csv` -- timings appended per run (rank, mode, kernel, size, ms, GFLOP/s, GB/s)
- `nsys_profiles/` -- Nsight Systems traces from `ncc_report.slurm`

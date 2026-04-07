#import "../lib.typ": *

#show: simple_report.with(
  title: [Parallel Scientific Computing: GPU & MPI],
  author: "Toby Davis (cltz62)",
  institution: "Durham University",
)

#let parse-results() = {
  let raw = csv("data/results.csv")
  let headers = raw.at(0)
  raw
    .slice(1)
    .map(row => {
      let d = (:)
      for (i, h) in headers.enumerate() {
        d.insert(h, row.at(i))
      }
      (
        rank: int(d.rank),
        num_ranks: int(d.num_ranks),
        mode: d.mode,
        kernel: d.kernel,
        N: int(d.N),
        global_N: int(d.global_N),
        M: int(d.M),
        Nmat: int(d.Nmat),
        K: int(d.K),
        ms: float(d.ms_gpu),
        gflops: float(d.GFLOPs),
        gbs: float(d.GBs),
      )
    })
    .filter(r => r.rank == 0)
}

#let summary = parse-results()

#let filter_vec(mode, np: 1) = {
  summary
    .filter(r => r.mode == mode and r.num_ranks == np)
    .sorted(key: r => r.global_N)
}

#let filter_gemm(kernel, np: 1) = {
  summary
    .filter(r => r.mode == "gemm" and r.kernel == kernel and r.num_ranks == np)
    .sorted(key: r => r.M)
}

#let filter_gemm_scaling(kernel, size) = {
  summary
    .filter(r => r.mode == "gemm" and r.kernel == kernel and r.M == size)
    .sorted(key: r => r.num_ranks)
}

#let colours = (
  axpy: rgb("#0072B2"),
  add: rgb("#E69F00"),
  copy: rgb("#009E73"),
  reduce: rgb("#CC79A7"),
  naive: rgb("#0072B2"),
  tiled: rgb("#E69F00"),
  optimised: rgb("#009E73"),
)

= Hardware & System

All benchmarks were performed on Durham University's NCC cluster with a single
CPU core and an NVIDIA 2080 Ti GPU with 11GB of GDDR6 VRAM and compute
capability 7.5. The 2080 Ti GPU has a peak memory bandwidth of
$616 "GBs"^(-1)$ and can perform $13.4 "TFLOPs"^(-1)$ in FP32 or
$107 "TFLOPs"^(-1)$ in FP16. All benchmarks were
compiled using CUDA 12.4 and Intel OneAPI MPI 2024.1.0.

= Vector Operations

== Vector Triad

#{
  let axpy_data = filter_vec("axpy", np: 1)
  let add_data = filter_vec("add", np: 1)
  let copy_data = filter_vec("copy", np: 1)
  let reduce_data = filter_vec("reduce", np: 1)

  figure(
    placement: none,
    scope: "column",
    lq.diagram(
      xlabel: [Vector Size ($N$)],
      ylabel: [Effective Bandwidth ($"GBs"^(-1)$)],
      width: 100%,
      height: 4.75cm,
      xscale: "log",
      ylim: (0, auto),
      legend: (position: right + bottom),

      lq.plot(
        axpy_data.map(r => r.global_N),
        axpy_data.map(r => r.gbs),
        mark: "o",
        label: [AXPY],
        stroke: colours.axpy,
      ),
      lq.plot(
        add_data.map(r => r.global_N),
        add_data.map(r => r.gbs),
        mark: "s",
        label: [ADD],
        stroke: colours.add,
      ),
      lq.plot(
        copy_data.map(r => r.global_N),
        copy_data.map(r => r.gbs),
        mark: "x",
        label: [COPY],
        stroke: colours.copy,
      ),
      lq.plot(
        reduce_data.map(r => r.global_N),
        reduce_data.map(r => r.gbs),
        mark: "^",
        label: [Reduction],
        stroke: colours.reduce,
      ),
    ),
    caption: [
      Effective memory bandwidth for vector operations at varying problem sizes
      with a single MPI rank.
    ],
  )
}<vector_triad_effective_bandwidth>

The vector triad operations shown in #ref(<vector_triad_effective_bandwidth>)
all show very similar memory bandwidth characteristics across different problem
sizes. For small $N$, there is not enough data to fully saturate the GPU memory
bus and CUDA cores, so memory and kernel dispatch latencies dominate and
bandwidth is low. As the problem size exceeds the number of CUDA cores on the
GPU, memory transfers can be overlapped with compute and memory bandwidth
becomes the bottleneck.

The 2080 Ti GPU has a maximum memory throughput of $616 "GBs"^(-1)$ and the
highest recorded throughput in the vector triad operations was
$553.3 "GBs"^(-1)$, suggesting that the kernels were not able to fully
saturate the memory bus, even at large problem sizes. Effective bandwidth is
computed as the total bytes transferred divided by the kernel time. AXPY and ADD
each read two arrays and write one ($3 N times 4$ bytes), COPY reads one and
writes one ($2 N times 4$ bytes), and the reduction reads a single array
($N times 4$ bytes).

#ref(<vector_triad_aggregate_mpi_bandwidth>) shows the total bandwidth across
all MPI ranks for a fixed problem size and varying numbers of MPI ranks.
Increasing the number of MPI ranks reduces the amount of data each rank must
process and reduces its effective bandwidth, as shown in
#ref(<vector_triad_effective_bandwidth>). While some of this computation can be
overlapped, the overhead of doing so reduces the available parallelism and leads
to degradation in performance.

#{
  let target_N = 50000000

  let scaling_data(mode) = {
    summary
      .filter(r => r.mode == mode and r.global_N == target_N)
      .sorted(key: r => r.num_ranks)
  }

  let axpy_s = scaling_data("axpy")
  let add_s = scaling_data("add")
  let copy_s = scaling_data("copy")
  let reduce_s = scaling_data("reduce")

  figure(
    placement: none,
    scope: "column",
    lq.diagram(
      xlabel: [Number of MPI Ranks],
      ylabel: [Aggregate Bandwidth ($"GBs"^(-1)$)],
      width: 100%,
      height: 4.75cm,
      xlim: (0.5, 4.5),
      xaxis: (ticks: (1, 2, 3, 4)),
      ylim: (0, auto),
      legend: (position: left + bottom),

      lq.plot(
        axpy_s.map(r => r.num_ranks),
        axpy_s.map(r => r.gbs * r.num_ranks),
        mark: "o",
        label: [AXPY],
        stroke: colours.axpy,
      ),
      lq.plot(
        add_s.map(r => r.num_ranks),
        add_s.map(r => r.gbs * r.num_ranks),
        mark: "s",
        label: [ADD],
        stroke: colours.add,
      ),
      lq.plot(
        copy_s.map(r => r.num_ranks),
        copy_s.map(r => r.gbs * r.num_ranks),
        mark: "x",
        label: [COPY],
        stroke: colours.copy,
      ),
      lq.plot(
        reduce_s.map(r => r.num_ranks),
        reduce_s.map(r => r.gbs * r.num_ranks),
        mark: "^",
        label: [Reduction],
        stroke: colours.reduce,
      ),
      lq.hlines(616, stroke: (dash: "dashed", paint: gray)),
    ),
    caption: [
      Aggregate memory bandwidth across MPI ranks for $N = 50 times 10^6$. Dashed
      line shows 2080 Ti peak ($616 "GBs"^(-1)$).
    ],
  )
}<vector_triad_aggregate_mpi_bandwidth>

// ═════════════════════════════════════════════════════════════════════════════
// SECTION: Reduction
// ═════════════════════════════════════════════════════════════════════════════

== Reduction

We implemented a tree-reduction algorithm with warp-level primitives to reduce
synchronisation overhead. We spawn at most 256 blocks, each containing 1024
threads. Each thread reduces a section of the problem into a local sum, which is
reduced via a warp-level reduction into another partial sum. Another pass of the
reduction algorithm on the partial sums is enough to compute the final result.
We load data as `float4` arrays to maximise memory bandwidth and ensure that all
threads in a warp are active to achieve the best possible performance, reaching
$546.1 "GBs"^(-1)$, matching the vector triad bandwidth.

IEEE 754 32-bit floating point numbers have a 23-bit mantissa, allowing for
relative error on the order of $1/2^23$. Additionally, we can approximate the
sum as a random walk of length $N$, so $sum_i x_i approx sqrt(N)$. Hence, we can
estimate the floating point error on a reduction of $N = 50 times 10^6$
elements uniformly centred around 0 as

#eq_no_num(
  $
    1 / (2^("mantissa")) times sqrt(N) approx 1.192 times 10^(-7) times 7071.067 approx 0.00084.
  $,
)

The maximum error observed with our reduction algorithm is $0.000732422$,
measured by comparing the GPU result against a double-precision CPU reference sum
across multiple random seeds. This confirms the algorithm is numerically stable
for large $N$.

// ═════════════════════════════════════════════════════════════════════════════
// SECTION: GEMM
// ═════════════════════════════════════════════════════════════════════════════

= General Matrix Multiply (GEMM)

In addition to the naive and tiled implementations, we implemented an optimised
F16 tiled WMMA algorithm which multiplies in F16 and accumulates into F32.
#ref(<gemm_throughput>) shows the performance of each GEMM algorithm over varying
matrix sizes.

The naive implementation assigns one thread per output element, so each thread
independently loads an entire row of $A$ and column of $B$ from global memory
with no data reuse between threads. This gives an arithmetic intensity of
roughly one floating point operation per four bytes loaded. Once the input
matrices exceed the L2 cache, every access goes to VRAM at full latency,
reducing performance by almost a third from its peak. The maximum observed error
was $0.000019$. For small matrices the L2 cache is sufficient to keep the GPU
fed and the naive implementation performs competitively with the tiled version.

The tiled implementation uses $32 times 32$ tiles loaded into shared memory
with `__syncthreads()` barriers between tile loads and computation to prevent
data races. Each thread accumulates its output element across all tiles along
the $K$ dimension, reusing data more efficiently and avoiding the penalties of
fetching from VRAM. This reaches a peak throughput of $1556.5 "GFLOPs"^(-1)$ with a
maximum error of $0.000019$.

By efficiently tiling the matrices, using tensor cores and reducing the
precision of the multiplications, the optimised algorithm is over 20 times
faster than the tiled version and yet has an error of just $0.011714$.

== Problem Size Scaling

#{
  let naive_data = filter_gemm("naive", np: 1)
  let tiled_data = filter_gemm("tiled", np: 1)
  let opt_data = filter_gemm("optimised", np: 1)

  figure(
    placement: none,
    scope: "column",
    lq.diagram(
      xlabel: [Matrix Size (M = N = K)],
      ylabel: [$"GFLOPs"^(-1)$],
      width: 100%,
      height: 4.75cm,
      yscale: "log",
      xscale: lq.scale.log(base: 2),
      ylim: (1, auto),
      legend: (position: left + top),

      lq.plot(
        naive_data.map(r => r.M),
        naive_data.map(r => r.gflops),
        mark: "o",
        label: [Naive],
        stroke: colours.naive,
      ),
      lq.plot(
        tiled_data.map(r => r.M),
        tiled_data.map(r => r.gflops),
        mark: "s",
        label: [Tiled],
        stroke: colours.tiled,
      ),
      lq.plot(
        opt_data.map(r => r.M),
        opt_data.map(r => r.gflops),
        mark: "x",
        label: [Optimised],
        stroke: colours.optimised,
      ),
    ),
    caption: [GEMM throughput ($"GFLOPs"^(-1)$) for each kernel at varying matrix sizes with a single MPI rank.],
  )
}<gemm_throughput>

== Strong Scaling

#{
  let target = 16384

  let naive_s = filter_gemm_scaling("naive", target)
  let tiled_s = filter_gemm_scaling("tiled", target)
  let opt_s = filter_gemm_scaling("optimised", target)

  figure(
    placement: none,
    scope: "column",
    lq.diagram(
      xlabel: [Number of MPI Ranks],
      ylabel: [Aggregate $"GFLOPs"^(-1)$],
      width: 100%,
      height: 4.75cm,
      xlim: (0.5, 4.5),
      xaxis: (ticks: (1, 2, 3, 4)),
      yscale: "log",
      ylim: (1, auto),
      legend: (position: right + bottom),

      lq.plot(
        naive_s.map(r => r.num_ranks),
        naive_s.map(r => r.gflops * r.num_ranks),
        mark: "o",
        label: [Naive],
        stroke: colours.naive,
      ),
      lq.plot(
        tiled_s.map(r => r.num_ranks),
        tiled_s.map(r => r.gflops * r.num_ranks),
        mark: "s",
        label: [Tiled],
        stroke: colours.tiled,
      ),
      lq.plot(
        opt_s.map(r => r.num_ranks),
        opt_s.map(r => r.gflops * r.num_ranks),
        mark: "x",
        label: [Optimised],
        stroke: colours.optimised,
      ),
    ),
    caption: [Aggregate $"GFLOPs"^(-1)$ for GEMM kernels as MPI ranks increase, showing scaling on a single GPU.],
  )
}<strong-scaling>

#{
  let target = 16384

  let naive_s = filter_gemm_scaling("naive", target)
  let tiled_s = filter_gemm_scaling("tiled", target)
  let opt_s = filter_gemm_scaling("optimised", target)

  figure(
    placement: none,
    scope: "column",
    lq.diagram(
      xlabel: [Number of MPI Ranks],
      ylabel: [Compute Time (ms)],
      width: 100%,
      height: 4.75cm,
      xlim: (0.5, 4.5),
      xaxis: (ticks: (1, 2, 3, 4)),
      yscale: "log",
      ylim: (1, auto),
      legend: (position: right + bottom),

      lq.plot(
        naive_s.map(r => r.num_ranks),
        naive_s.map(r => r.ms),
        mark: "o",
        label: [Naive],
        stroke: colours.naive,
      ),
      lq.plot(
        tiled_s.map(r => r.num_ranks),
        tiled_s.map(r => r.ms),
        mark: "s",
        label: [Tiled],
        stroke: colours.tiled,
      ),
      lq.plot(
        opt_s.map(r => r.num_ranks),
        opt_s.map(r => r.ms),
        mark: "x",
        label: [Optimised],
        stroke: colours.optimised,
      ),
    ),
    caption: [Rank 0 compute time for GEMM kernels as MPI ranks increase. Each rank computes fewer rows, but GPU contention limits speedup.],
  )
}<strong-scaling-time>

#{
  let opt_scaling = summary
    .filter(r => r.mode == "gemm" and r.kernel == "optimised" and r.M == 16384)
    .sorted(key: r => r.num_ranks)

  figure(
    placement: none,
    scope: "column",
    table(
      columns: 5,
      align: (left, right, right, right, right),
      table.header[Ranks][Compute (ms)][Comm (ms)][Total (ms)][Comm %],
      ..opt_scaling
        .map(r => {
          let total = r.ms + r.gbs
          let pct = r.gbs / total * 100
          (
            [#r.num_ranks],
            [#calc.round(r.ms, digits: 1)],
            [#calc.round(r.gbs, digits: 1)],
            [#calc.round(total, digits: 1)],
            [#calc.round(pct, digits: 1)%],
          )
        })
        .flatten(),
    ),
    caption: [Compute and communication time for the optimised GEMM kernel at $M = N = K = 16384$.],
  )
}<scaling-table>

For a fixed problem size of $M = N = K = 16384$,
#ref(<strong-scaling>) and #ref(<strong-scaling-time>) show that the GPU
throughput and compute time are not dependent on the number of MPI ranks used.
This is because the GPU is fully occupied by the computation and there is no way
to overlap the per-rank computation, so they run near-sequentially and the
runtime is unaffected.

Communication time was measured separately via an MPI barrier before each kernel
launch. In the worst case, communication jitter takes 26.5% of the total time.
On average, however, communication accounts for only 7.93% of the total runtime. This suggests
that MPI communication is not the bottleneck, and that a more optimised GEMM
implementation would lead to consistently better performance.

The factor limiting the scaling of the GEMM kernel is the GPU's compute
throughput and memory bandwidth. If distinct GPUs could be assigned per rank,
the throughput could be expected to scale linearly with the number of ranks due
to the almost negligible communication overhead.

== Profiled Results

#figure(
  placement: none,
  scope: "column",
  image("data/NVIDIA Nsight Systems 2026-04-07 at 11.42.02@2x.png"),
  caption: [Profiling a 2-rank benchmark in Nsight Systems],
)<profile>

The profiling results shown in #ref(<profile>) report that only 5.4% of the time
is spent in `convert_f32_to_f16`, with the remaining 94.6% of the time being
spent in the WMMA GEMM kernel. It is clear that further optimisation efforts
should be focused on the compute-intensive GEMM kernel rather than the simple
and fast data cast/copy. The profile also shows that memory copies dominate the
runtime of the program and therefore should be avoided at all costs.

The profile also shows that very minimal time is spent in MPI
communication, which is consistent with the benchmark results above. GPU idle
time between kernel launches is negligible, confirming that the device is
compute-bound. Each rank launches both the F32-to-F16 conversion and the WMMA
GEMM, with launches separated only by the implicit synchronisation between them.

Given more time, we would write inline PTX to better overlap memory loads with
compute and to gain finer control over tile sizes. On newer GPU architectures
(sm_80+), WMMA supports F32 operands, which would eliminate the precision
penalty of F16 while retaining tensor core throughput.


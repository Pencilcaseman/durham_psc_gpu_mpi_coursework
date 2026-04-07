#import "../lib.typ": *

#show: simple_report.with(
  title: [Parallel Scientific Computing: GPU & MPI],
  author: "Toby Davis (cltz62)",
  institution: "Durham University",
  // bibliography: bibliography("refs.bib"),
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
  axpy: rgb("#e41a1c"),
  add: rgb("#377eb8"),
  copy: rgb("#4daf4a"),
  reduce: rgb("#984ea3"),
  naive: rgb("#e41a1c"),
  tiled: rgb("#377eb8"),
  optimised: rgb("#4daf4a"),
)

= Hardware

All benchmarks were performed on Durham University's NCC cluster with a single
CPU core and an NVIDIA 2080 Ti GPU with 11GB of GDDR6 VRAM and compute
capability 7.5. The 2080 Ti GPU has a peak memory bandwidth of 616 GB/s and can
perform 13.4 TFLOP/s in FP32 or 107 TFLOP/s in FP16. All benchmarks were
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
      ylabel: [Effective Bandwidth (GB/s)],
      width: 100%,
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
        label: [Reduce],
        stroke: colours.reduce,
      ),
    ),
    caption: [
      Effective memory bandwidth for vector operations at varying problem sizes
      with a single MPI rank.
    ],
  )
}<vector_triad_effective_bandwidth>

The vector triad operations `AXPY`, `ADD` and `COPY` all show very similar
memory bandwidth characteristics across different problem sizes. For small
$N$, there is not enough data to fully saturate the GPU memory bus and CUDA
cores, so memory latencies dominate and bandwidth is low. As the problem size
exceeds the number of CUDA cores on the GPU, memory transfers can be overlapped
with compute and memory bandwidth becomes the bottleneck.

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
      ylabel: [Aggregate Bandwidth (GB/s)],
      width: 100%,
      xlim: (0.5, 4.5),
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
        label: [Reduce],
        stroke: colours.reduce,
      ),
      lq.hlines(616, stroke: (dash: "dashed", paint: gray)),
    ),
    caption: [
      Aggregate memory bandwidth across MPI ranks for N=$50 times 10^6$. Dashed
      line shows 2080 Ti peak (616 GB/s).
    ],
  )
}<vector_triad_aggregate_mpi_bandwidth>


// ═════════════════════════════════════════════════════════════════════════════
// SECTION: Reduction
// ═════════════════════════════════════════════════════════════════════════════

== Reduction

// ═════════════════════════════════════════════════════════════════════════════
// SECTION: GEMM
// ═════════════════════════════════════════════════════════════════════════════

= General Matrix Multiply (GEMM)

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
      ylabel: [GFLOP/s],
      width: 100%,
      yscale: "log",
      xscale: "log",
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
    caption: [GEMM throughput (GFLOP/s) for each kernel at varying matrix sizes with a single MPI rank.],
  )
}

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
      ylabel: [Aggregate GFLOP/s],
      width: 100%,
      xlim: (0.5, 4.5),
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
    caption: [Aggregate GFLOP/s for GEMM kernels as MPI ranks increase, showing scaling on a single GPU.],
  )
}

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
      xlim: (0.5, 4.5),
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
}

== Profiled Results

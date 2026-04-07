#import "../lib.typ": *

#show: simple_report.with(
  title: [Parallel Scientific Computing: GPU & MPI],
  author: "Toby Davis (cltz62)",
  institution: "Durham University",
  bibliography: bibliography("refs.bib"),
)

// ── Data loading and helpers ────────────────────────────────────────────────

#{
  // Parse CSV into array of dicts, keeping only rank 0 rows
  let raw = csv("data/results.csv")
  let headers = raw.at(0)
  let all_rows = raw.slice(1).map(row => {
    let d = (:)
    for (i, h) in headers.enumerate() {
      d.insert(h, row.at(i))
    }
    // Parse numeric fields
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

  // Only use rank 0 data for reporting
  let summary = all_rows.filter(r => r.rank == 0)

  // Helper: filter and sort
  let filter_vec(mode, np: 1) = {
    summary
      .filter(r => r.mode == mode and r.num_ranks == np)
      .sorted(key: r => r.global_N)
  }

  let filter_gemm(kernel, np: 1) = {
    summary
      .filter(r => r.mode == "gemm" and r.kernel == kernel and r.num_ranks == np)
      .sorted(key: r => r.M)
  }

  let filter_gemm_scaling(kernel, size) = {
    summary
      .filter(r => r.mode == "gemm" and r.kernel == kernel and r.M == size)
      .sorted(key: r => r.num_ranks)
  }

  let colours = (
    axpy: rgb("#e41a1c"),
    add: rgb("#377eb8"),
    copy: rgb("#4daf4a"),
    reduce: rgb("#984ea3"),
    naive: rgb("#e41a1c"),
    tiled: rgb("#377eb8"),
    optimised: rgb("#4daf4a"),
  )

  // ══════════════════════════════════════════════════════════════════════════
  // SECTION: Vector Triad
  // ══════════════════════════════════════════════════════════════════════════

  [= Vector Triad]

  // --- Bandwidth vs N (NP=1) ---
  {
    let axpy_data = filter_vec("axpy", np: 1)
    let add_data = filter_vec("add", np: 1)
    let copy_data = filter_vec("copy", np: 1)
    let reduce_data = filter_vec("reduce", np: 1)

    [
      #figure(
        placement: none,
        scope: "column",
        lq.diagram(
          title: [Bandwidth vs Problem Size (NP=1)],
          xlabel: [Vector Size (N)],
          ylabel: [Effective Bandwidth (GB/s)],
          width: 100%,
          xscale: "log",

          lq.plot(
            axpy_data.map(r => r.global_N),
            axpy_data.map(r => r.gbs),
            mark: "o", label: [AXPY], stroke: colours.axpy,
          ),
          lq.plot(
            add_data.map(r => r.global_N),
            add_data.map(r => r.gbs),
            mark: "s", label: [ADD], stroke: colours.add,
          ),
          lq.plot(
            copy_data.map(r => r.global_N),
            copy_data.map(r => r.gbs),
            mark: "x", label: [COPY], stroke: colours.copy,
          ),
          lq.plot(
            reduce_data.map(r => r.global_N),
            reduce_data.map(r => r.gbs),
            mark: "^", label: [Reduce], stroke: colours.reduce,
          ),
        ),
        caption: [Effective memory bandwidth for vector operations and reduction at varying problem sizes with a single MPI rank.],
      )
    ]
  }

  // --- Bandwidth strong scaling (N=50M) ---
  {
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

    [
      #figure(
        placement: none,
        scope: "column",
        lq.diagram(
          title: [Aggregate Bandwidth vs MPI Ranks (N=50M)],
          xlabel: [Number of MPI Ranks],
          ylabel: [Aggregate Bandwidth (GB/s)],
          width: 100%,
          xlim: (0.5, 4.5),

          lq.plot(
            axpy_s.map(r => r.num_ranks),
            axpy_s.map(r => r.gbs * r.num_ranks),
            mark: "o", label: [AXPY], stroke: colours.axpy,
          ),
          lq.plot(
            add_s.map(r => r.num_ranks),
            add_s.map(r => r.gbs * r.num_ranks),
            mark: "s", label: [ADD], stroke: colours.add,
          ),
          lq.plot(
            copy_s.map(r => r.num_ranks),
            copy_s.map(r => r.gbs * r.num_ranks),
            mark: "x", label: [COPY], stroke: colours.copy,
          ),
          lq.plot(
            reduce_s.map(r => r.num_ranks),
            reduce_s.map(r => r.gbs * r.num_ranks),
            mark: "^", label: [Reduce], stroke: colours.reduce,
          ),
          lq.hlines(616, stroke: (dash: "dashed", paint: gray)),
        ),
        caption: [Aggregate bandwidth across MPI ranks for N=50M. Dashed line shows 2080 Ti peak (616 GB/s).],
      )
    ]
  }

  // ══════════════════════════════════════════════════════════════════════════
  // SECTION: Reduction
  // ══════════════════════════════════════════════════════════════════════════

  [= Reduction]

  // ══════════════════════════════════════════════════════════════════════════
  // SECTION: GEMM
  // ══════════════════════════════════════════════════════════════════════════

  [= General Matrix Multiply (GEMM)]

  [== Problem Size Scaling]

  // --- GFLOP/s vs matrix size (NP=1) ---
  {
    let naive_data = filter_gemm("naive", np: 1)
    let tiled_data = filter_gemm("tiled", np: 1)
    let opt_data = filter_gemm("optimised", np: 1)

    [
      #figure(
        placement: none,
        scope: "column",
        lq.diagram(
          title: [GEMM Performance vs Problem Size (NP=1)],
          xlabel: [Matrix Size (M = N = K)],
          ylabel: [GFLOP/s],
          width: 100%,
          yscale: "log",
          xscale: "log",

          lq.plot(
            naive_data.map(r => r.M),
            naive_data.map(r => r.gflops),
            mark: "o", label: [Naive], stroke: colours.naive,
          ),
          lq.plot(
            tiled_data.map(r => r.M),
            tiled_data.map(r => r.gflops),
            mark: "s", label: [Tiled], stroke: colours.tiled,
          ),
          lq.plot(
            opt_data.map(r => r.M),
            opt_data.map(r => r.gflops),
            mark: "x", label: [Optimised], stroke: colours.optimised,
          ),
        ),
        caption: [GEMM throughput (GFLOP/s) for each kernel at varying matrix sizes with a single MPI rank.],
      )
    ]
  }

  [== Strong Scaling]

  // --- GEMM strong scaling: aggregate GFLOP/s vs NP at largest common size ---
  {
    let target = 16384

    let naive_s = filter_gemm_scaling("naive", target)
    let tiled_s = filter_gemm_scaling("tiled", target)
    let opt_s = filter_gemm_scaling("optimised", target)

    [
      #figure(
        placement: none,
        scope: "column",
        lq.diagram(
          title: [GEMM Strong Scaling: Aggregate Throughput (#str(target) #sym.times #str(target))],
          xlabel: [Number of MPI Ranks],
          ylabel: [Aggregate GFLOP/s],
          width: 100%,
          xlim: (0.5, 4.5),

          lq.plot(
            naive_s.map(r => r.num_ranks),
            naive_s.map(r => r.gflops * r.num_ranks),
            mark: "o", label: [Naive], stroke: colours.naive,
          ),
          lq.plot(
            tiled_s.map(r => r.num_ranks),
            tiled_s.map(r => r.gflops * r.num_ranks),
            mark: "s", label: [Tiled], stroke: colours.tiled,
          ),
          lq.plot(
            opt_s.map(r => r.num_ranks),
            opt_s.map(r => r.gflops * r.num_ranks),
            mark: "x", label: [Optimised], stroke: colours.optimised,
          ),
        ),
        caption: [Aggregate GFLOP/s for GEMM kernels as MPI ranks increase, showing scaling on a single GPU.],
      )
    ]
  }

  // --- GEMM strong scaling: compute time vs NP ---
  {
    let target = 16384

    let naive_s = filter_gemm_scaling("naive", target)
    let tiled_s = filter_gemm_scaling("tiled", target)
    let opt_s = filter_gemm_scaling("optimised", target)

    [
      #figure(
        placement: none,
        scope: "column",
        lq.diagram(
          title: [GEMM Rank 0 Compute Time (#str(target) #sym.times #str(target))],
          xlabel: [Number of MPI Ranks],
          ylabel: [Compute Time (ms)],
          width: 100%,
          xlim: (0.5, 4.5),

          lq.plot(
            naive_s.map(r => r.num_ranks),
            naive_s.map(r => r.ms),
            mark: "o", label: [Naive], stroke: colours.naive,
          ),
          lq.plot(
            tiled_s.map(r => r.num_ranks),
            tiled_s.map(r => r.ms),
            mark: "s", label: [Tiled], stroke: colours.tiled,
          ),
          lq.plot(
            opt_s.map(r => r.num_ranks),
            opt_s.map(r => r.ms),
            mark: "x", label: [Optimised], stroke: colours.optimised,
          ),
        ),
        caption: [Rank 0 compute time for GEMM kernels as MPI ranks increase. Each rank computes fewer rows, but GPU contention limits speedup.],
      )
    ]
  }

  [== Profiled Results]
}


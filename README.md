# CUDA K-Means Clustering

K-means clustering in CUDA C. Three kernels per iteration, a two-stage reduction, and a
single shared-memory buffer split into six regions. Written in 2015 for a graduate
HPC course at Western University, and benchmarked against MPI, OpenMP and sequential
implementations.

<p align="center">
  <img src="assets/pipeline.svg" width="880" alt="CUDA k-means execution model">
</p>

*99,968 points in 3D, 781 blocks of 128 threads, iterations 1, 3, 8 and 31. Each block
is shaded by how many of its points changed cluster. The diagram is produced by running
the algorithm; `tools/verify-trace.mjs` checks its numbers against `src/` first.*

**[Interactive version](https://danielberhane.github.io/cuda-kmeans/)**: the same
algorithm on the same data, in the browser. Change K, block size or threshold and step
through iterations.

## Results

1D data, 100,000 points. Baselines are the Northwestern parallel k-means package
(Wei-keng Liao, Serban Giuroiu). `MyCUDA` is this repository.

<p align="center">
  <img src="presentation/figures/time-vs-clusters-1d-100k.png" width="860" alt="Computation time vs. number of clusters">
</p>

| K   | Sequential | OpenMP  | MPI     | **MyCUDA** | CUDA (ref) |
|-----|-----------|---------|---------|------------|------------|
| 2   | 0.06 s    | 0.10 s  | 0.10 s  | **0.16 s** | 0.19 s     |
| 20  | 0.39 s    | 0.51 s  | 0.19 s  | **0.19 s** | 0.19 s     |
| 50  | 1.02 s    | 1.17 s  | 1.20 s  | **0.33 s** | 0.19 s     |
| 100 | 1.93 s    | 1.70 s  | 1.10 s  | **0.37 s** | 0.22 s     |

CPU time grows with K. GPU time is nearly flat. At K=100 this implementation is 5×
faster than sequential.

At K=2 it is 4–10× slower than sequential, and takes the same 0.17 s for 10,000 points
as for 100,000. That is fixed cost: context creation, transfers, and a blocking copy
each iteration. There is not enough arithmetic at K=2 to cover it.

**Per-kernel bandwidth** (n=100,000, k=100, d=1, 128 threads/block):

| Kernel | Read | Written | Time | Effective BW |
|--------|------|---------|------|--------------|
| `find_nearest_cluster`    | 801,400 B | 1,024,800 B | 0.24 ms | 6 GB/s\* |
| `reduce_coord_clusters`   | 627,000 B | 800 B       | 0.18 ms | 3.48 GB/s |
| `reduce_cluster_changed`  | 3,124 B   | 6,248 B     | 0.12 ms | 0.078 GB/s |

`reduce_cluster_changed` runs as one thread block and takes 22% of the kernel time to
move 0.5% of the bytes. Spreading it across blocks is the next improvement.

\* Inconsistent with its own inputs, which give 7.61 GB/s. See
[`presentation/BENCHMARKS.md`](presentation/BENCHMARKS.md).

## Design

The host transposes the input from `[N][D]` to `[D][N]` once, so each warp reads a
coordinate as one coalesced access. Initial centroids are the first K points. Each
iteration then launches three kernels:

1. **`find_nearest_cluster`**, one thread per point. Loads its tile of points and all
   centroids into shared memory, assigns each point, counts changes with a tree
   reduction, and accumulates per-block sums of coordinates for each cluster.
2. **`reduce_cluster_changed`**, one block. Sums the per-block change counts. The host
   reads the result to test convergence.
3. **`reduce_coord_clusters`**, one block per cluster. Sums the per-block coordinate
   partials, two blocks' worth at a time, by tree reduction.

The host divides the sums to get new centroids and loops until fewer than `threshold`
of the points change cluster, or 500 iterations.

Limits, all from the fixed block size of 128:

- K × D must not exceed 128; the centroid loads and writes use one thread per value.
- Shared memory for kernel 1 is `128·D·4 + 2·K·D·4 + 4·K + 256` bytes.
- `reduce_coord_clusters` is launched with N/128 blocks but only K of them work.

## Volta and later

The code assumed warp-synchronous execution: the 32 threads of a warp advancing in
lockstep, which every NVIDIA GPU guaranteed through Pascal. Four reductions unrolled
their last warp with no synchronisation. Volta (2017) dropped the guarantee. On an
A100 the reductions lose updates, the change count comes out low, and the loop exits
after 6 iterations instead of 31.

Fixes in this repository:

- The four reductions synchronise at every step.
- `reduce_cluster_changed` folds strided slices instead of mapping one thread per block.
  The old mapping needed more than 1,024 threads past N=131,072 and failed to launch.
- `s_membership` is written for every thread. It was set only for points that changed
  cluster, so from iteration 2 the centroid sums read uninitialised shared memory.

The scaling sweep now runs to 5,000,000 points on an A100. **Correctness on Volta and
later is not yet confirmed**: an A100 run still disagrees with the CPU baseline on
iteration count, and the benchmark [withholds timings](bench/README.md) until they
match. The figures above are from 2015 hardware.

## Build and run

Requires the CUDA Toolkit and a CUDA-capable GPU.

```bash
make                          # cuda_kmeans
make GENCODE="-arch=sm_80"    # set the architecture before benchmarking
make cuda_kmeans_bench        # per-kernel timings
make bench-tools              # CPU baseline and data generator, no CUDA needed

./cuda_kmeans <K> <D> <N> <threshold> <input_file> [--csv]
./cuda_kmeans 4 3 99968 0.001 data/points_3d.txt
```

Writes `<input>.cluster_centres` and `<input>.membership`. `--csv` prints one row
instead. The bundled datasets have no cluster structure; they measure throughput. For
the full benchmark see [`bench/README.md`](bench/README.md).

## Credits

Kernels, host driver and benchmark scripts: Daniel Berhane Araya. Original design note
in [`presentation/kernel-design-notes.txt`](presentation/kernel-design-notes.txt).

`cuda_io.cu`, `cuda_wtime.cu`, `kmeans.h` and the Makefile layout are adapted from the
MIT-licensed Northwestern package by Wei-keng Liao and Serban Giuroiu, which also
supplied the comparison baselines.

MIT License. See [LICENSE](LICENSE).

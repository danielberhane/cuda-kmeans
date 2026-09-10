# Benchmarks from the original 2015 presentation

Transcribed from `k-means-cuda-2015.pptx` (Daniel Berhane, 3 May 2015, Western
University, Department of Electrical and Computer Engineering). Figures extracted to
`figures/`.

The slides report figures only as charts -- the deck contains no embedded
chart data, so the values below are read off the plots and are approximate
where noted. Hardware and CUDA version are not stated anywhere in the deck.

## 1. Computation time vs. number of clusters

1D data, 100,000 points. Five implementations compared:
MPI, OpenMP, Sequential, CUDA (reference implementation), MyCUDA (this repo).

![time vs clusters](figures/time-vs-clusters-1d-100k.png)

Approximate values read from the chart:

| K   | Sequential | OpenMP | MPI  | MyCUDA | CUDA (ref) |
|-----|-----------|--------|------|--------|------------|
| 2   | ~0.06 s   | ~0.10 s| ~0.10 s | ~0.16 s | ~0.19 s |
| 20  | ~0.39 s   | ~0.51 s| ~0.19 s | ~0.19 s | ~0.19 s |
| 50  | ~1.02 s   | ~1.17 s| ~1.20 s | ~0.33 s | ~0.19 s |
| 80  | ~1.33 s   | ~1.70 s| ~1.30 s | ~0.35 s | ~0.19 s |
| 100 | ~1.93 s   | ~1.70 s| ~1.10 s | ~0.37 s | ~0.22 s |

Both GPU implementations are essentially flat in K while every CPU
implementation grows roughly linearly. At K=100 MyCUDA is about **5x faster
than sequential**; below roughly K=20 it is slower.

## 2. Computation time vs. data size at K=2

1D data, cluster size fixed at 2, data size swept from 10,000 to 100,000.

![time vs data size](figures/time-vs-datasize-1d-k2.png)

| N       | Sequential | MyCUDA  |
|---------|-----------|---------|
| 10,000  | ~0.072 s  | ~0.165 s |
| 30,000  | ~0.013 s  | ~0.165 s |
| 60,000  | ~0.005 s  | ~0.170 s |
| 80,000  | ~0.040 s  | ~0.186 s |
| 100,000 | ~0.046 s  | ~0.172 s |

MyCUDA is flat at ~0.17 s across the whole range -- that is fixed overhead
(context creation and host/device transfer), not computation. The CPU is
4-10x faster here. This is the measured basis for the deck's conclusion:

> "For small cluster size the overhead of data copying between device and
> host outweighs the parallelism advantages"

## 3. Effective bandwidth per kernel

Measured at n=100,000, k=100, d=1, block size Z=128 -- the same 128 threads
per block the code still uses.

`BW_effective = (bytes_read + bytes_written) / (t * 1e9)`

| Kernel | Read | Written | t | Effective BW |
|--------|------|---------|---|--------------|
| 1. `find_nearest_cluster` | 801,400 B | 1,024,800 B | 0.24 ms | 6 GB/s *(as stated)* |
| 2. `reduce_coord_clusters` | 627,000 B | 800 B | 0.18 ms | 3.48 GB/s |
| 3. `reduce_cluster_changed` | 3,124 B | 6,248 B | 0.12 ms | 0.078 GB/s |

**Discrepancy, unresolved:** kernel 1's stated 6 GB/s does not follow from
its own stated inputs -- (801,400 + 1,024,800) / (0.24e-3 * 1e9) = 7.61 GB/s.
Kernels 2 and 3 both reproduce exactly from their stated inputs, so the
6 GB/s figure is likely a slip in either the byte count or the elapsed time.
It should be re-measured before being quoted anywhere.

**The interesting result** is kernel 3: it spends 22% of the per-iteration
kernel time (0.12 of 0.54 ms) moving 0.5% of the bytes. That is the
`reduce_cluster_changed<<<1, numReductionThreads>>>` launch -- a single
thread block, which leaves nearly every SM on the device idle.

## What is missing

- Hardware: GPU model, CPU model, node type are not recorded anywhere.
- Only 1D data is benchmarked; the repo ships 3D and 10D datasets too.
- Iteration counts are not reported, so it cannot be confirmed that the
  compared implementations converged identically.
- No error bars or repeat counts.

`bench/run_bench.slurm` addresses all four for the re-run.

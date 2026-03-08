# CUDA K-Means Clustering

A GPU-accelerated k-means clustering implementation in CUDA C, designed for large datasets with arbitrary dimensionality.

## How It Works

K-means partitions N data points into K clusters by repeatedly:
1. **Assigning** each point to the nearest cluster center (Euclidean distance)
2. **Updating** each cluster center to the mean of its assigned points
3. **Repeating** until fewer than a threshold fraction of points change clusters

This implementation parallelizes both steps on the GPU using three CUDA kernels:
- `find_nearest_cluster` — computes nearest cluster for each point and accumulates per-block coordinate sums using shared memory
- `reduce_coord_clusters` — reduces the per-block sums across all blocks to produce global cluster sums
- `reduce_cluster_changed` — counts how many points changed clusters to check convergence

## Build

Requires the NVIDIA CUDA Toolkit and a CUDA-capable GPU.

```bash
make            # builds the cuda_kmeans executable
make clean      # removes build artifacts
```

## Usage

```bash
./cuda_kmeans <num_clusters> <num_dimensions> <num_points> <threshold> <input_file>
```

| Argument | Description |
|----------|-------------|
| `num_clusters` | Number of clusters (K) |
| `num_dimensions` | Number of dimensions per point (auto-detected from file) |
| `num_points` | Number of data points |
| `threshold` | Convergence threshold (fraction of points that changed, e.g. 0.001) |
| `input_file` | Path to the input data file |

### Examples

```bash
./cuda_kmeans 4 3 99968 0.001 data/points_3d.txt
./cuda_kmeans 4 10 99968 0.001 data/points_10d.txt
```

### Input Format

One point per line. First column is the index (ignored), remaining columns are coordinates:
```
1 61.015527 20.459306 32.295319
2 63.265007 76.450910 14.535881
```

### Output

Two files are produced alongside the input file:
- `<input>.cluster_centres` — the K cluster center coordinates
- `<input>.membership` — the cluster ID assigned to each point

## Project Structure

```
├── src/
│   ├── cuda_kmeans.cu    # CUDA kernels (assignment, reduction, convergence)
│   ├── cuda_main.cu      # Main driver
│   ├── cuda_io.cu        # File I/O
│   ├── cuda_wtime.cu     # Wall-clock timer
│   └── kmeans.h          # Shared header and utility macros
├── data/
│   ├── points_1d.txt     # 99,968 points, 1 dimension
│   ├── points_3d.txt     # 99,968 points, 3 dimensions
│   ├── points_10d.txt    # 99,968 points, 10 dimensions
│   └── sample_output/    # Reference outputs for verification
├── Makefile
├── LICENSE
└── README.md
```

## Acknowledgments

The I/O utilities (`cuda_io.cu`, `cuda_wtime.cu`), header (`kmeans.h`), and original Makefile structure are based on code by **Wei-keng Liao** (Northwestern University) and **Serban Giuroiu**, released under the MIT License.

The CUDA k-means kernels and main driver were written by **Daniel Berhane Araya**.

## License

MIT License. See [LICENSE](LICENSE) for full details.

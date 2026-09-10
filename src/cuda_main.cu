/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*   File:         cuda_main.cu                                            */
/*   Description:  Main driver for CUDA k-means clustering                 */
/*                                                                         */
/*   Author:  Daniel Berhane Araya                                         */
/*   Copyright (c) 2016 Daniel Berhane Araya                               */
/*                                                                         */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

// MIT License - see LICENSE file for details

#include <stdio.h>
#include <stdlib.h>
#include <string.h>     /* strtok() */
#include <sys/types.h>  /* open() */
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cuda_runtime.h>

#include "kmeans.h"

/*---< main() >-------------------------------------------------------------*/
int main(int argc, char **argv) {

           int     is_output_timing;
           int     numClusters, numCoords, numObjs;
           int    *membership;    /* [numObjs] */
           char   *filename;
           float **objects;       /* [numObjs][numCoords] data objects */
           float **clusters;      /* [numClusters][numCoords] cluster center */
           float   threshold;
           double  timing, io_timing, clustering_timing;
           int     loop_iterations = 500;
           int a;
           kmeans_perf_t perf;
           int emit_csv = 0;

  if (argc < 6) {
    printf("Too few arguments\n");
    printf("Usage: ./cuda_kmeans <num_clusters> <num_dimensions> <num_points> "
           "<threshold> <input_file> [--csv]\n");
    return 1;
  }

  /* --csv emits one machine-readable row on stdout for the benchmark sweep,
     in place of the human-readable summary. */
  if (argc > 6 && strcmp(argv[6], "--csv") == 0) emit_csv = 1;

  memset(&perf, 0, sizeof(perf));

  /* cluster size, dimension, data-size and counter to keep track the number of iterations
     before converging
  */
  numClusters = atoi(argv[1]);
  numCoords = atoi(argv[2]);
  numObjs = atoi(argv[3]);
  threshold = atof(argv[4]);
  
  filename = argv[5];
  threshold = (float)threshold;
  is_output_timing = 1;

  if (is_output_timing) io_timing = wtime();

  objects = file_read(filename , &numObjs, &numCoords);
   if (objects == NULL) exit(1);

 if (is_output_timing) {
        timing            = wtime();
        io_timing         = timing - io_timing;
        clustering_timing = timing;
    }
    membership = (int*) malloc(numObjs * sizeof(int));
    assert(membership != NULL);

    a = cuda_kmeans(objects, numCoords, numObjs, numClusters, threshold, membership, &loop_iterations, &clusters, &perf);

    if (a != 0) {
        err("cuda_kmeans() failed with status %d\n", a);
    }

    free(objects[0]);
    free(objects);

    if (is_output_timing) {
        timing            = wtime();
        clustering_timing = timing - clustering_timing;
    }

    /* Benchmark sweeps run this hundreds of times; writing two files of
       numObjs lines on each run would dominate the wall clock and litter the
       data directory. */
    if (!emit_csv) {
        file_write(filename, numClusters, numObjs, numCoords, clusters,
                   membership);
    }

    free(membership);
    free(clusters[0]);
    free(clusters);

    io_timing += wtime() - timing;

    if (emit_csv) {
        /* Column order must match bench/run_bench.slurm's header line. */
        printf("%s,%d,%d,%d,%g,%d,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f,%.4f,%d,%d,%d,%zu,\"%s\",%d,%d.%d\n",
               filename, numObjs, numCoords, numClusters, threshold,
               perf.iterations,
               perf.clustering_sec, perf.transfer_sec, clustering_timing, io_timing,
               perf.find_nearest_ms, perf.reduce_coord_ms, perf.reduce_changed_ms,
               perf.num_blocks, perf.threads_per_block, perf.reduction_threads,
               perf.shared_bytes,
               perf.gpu_name, perf.sm_count, perf.cc_major, perf.cc_minor);
    } else {
        printf("\n*** Kmeans (CUDA version) Output ***\n\n");
        printf("Input file:        %s\n", filename);
        printf("numObjs          = %d\n", numObjs);
        printf("numCoords        = %d\n", numCoords);
        printf("numClusters      = %d\n", numClusters);
        printf("threshold        = %g\n", threshold);
        printf("iterations       = %d\n", perf.iterations);
        printf("\n");
        printf("grid             = %d blocks x %d threads\n",
               perf.num_blocks, perf.threads_per_block);
        printf("reduction kernel = 1 block x %d threads\n", perf.reduction_threads);
        printf("shared mem/block = %zu bytes\n", perf.shared_bytes);
        printf("device           = %s (%d SMs, cc %d.%d)\n",
               perf.gpu_name, perf.sm_count, perf.cc_major, perf.cc_minor);
        printf("\n");
        printf("I/O time         = %10.4f sec\n", io_timing);
        printf("Clustering time  = %10.4f sec  (convergence loop)\n", perf.clustering_sec);
        printf("  of which H2D/D2H %10.4f sec\n", perf.transfer_sec);
        printf("End-to-end GPU   = %10.4f sec  (incl. alloc, transpose, teardown)\n",
               clustering_timing);
#ifdef BENCH_KERNEL_BREAKDOWN
        printf("\n  per-kernel totals (breakdown build; syncs inflate the total)\n");
        printf("  find_nearest_cluster   = %10.4f ms\n", perf.find_nearest_ms);
        printf("  reduce_coord_clusters  = %10.4f ms\n", perf.reduce_coord_ms);
        printf("  reduce_cluster_changed = %10.4f ms\n", perf.reduce_changed_ms);
#endif
        printf("\n");
    }

    return(0);


}







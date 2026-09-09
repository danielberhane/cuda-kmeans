/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*   File:         kmeans.h   (an OpenMP version)                            */
/*   Description:  header file for a simple k-means clustering program       */
/*                                                                           */
/*   Author:  Wei-keng Liao                                                  */
/*            ECE Department Northwestern University                         */
/*            email: wkliao@ece.northwestern.edu                             */
/*   Copyright, 2005, Wei-keng Liao                                          */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

// Copyright (c) 2005 Wei-keng Liao
// Copyright (c) 2011 Serban Giuroiu
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// -----------------------------------------------------------------------------

#ifndef _H_KMEANS
#define _H_KMEANS

#include <assert.h>

#define msg(format, ...) do { fprintf(stderr, format, ##__VA_ARGS__); } while (0)
#define err(format, ...) do { fprintf(stderr, format, ##__VA_ARGS__); exit(1); } while (0)

#define malloc2D(name, xDim, yDim, type) do {               \
    name = (type **)malloc(xDim * sizeof(type *));          \
    assert(name != NULL);                                   \
    name[0] = (type *)malloc(xDim * yDim * sizeof(type));   \
    assert(name[0] != NULL);                                \
    for (size_t i = 1; i < xDim; i++)                       \
        name[i] = name[i-1] + yDim;                         \
} while (0)

#ifdef __CUDACC__
inline void checkCuda(cudaError_t e) {
    if (e != cudaSuccess) {
        // cudaGetErrorString() isn't always very helpful. Look up the error
        // number in the cudaError enum in driver_types.h in the CUDA includes
        // directory for a better explanation.
        err("CUDA Error %d: %s\n", e, cudaGetErrorString(e));
    }
}

inline void checkLastCudaError() {
    checkCuda(cudaGetLastError());
}
#endif

/* -----------------------------------------------------------------------------
 * Benchmark instrumentation.
 *
 * Filled in by cuda_kmeans().  Kernel breakdown fields are only populated when
 * built with -DBENCH_KERNEL_BREAKDOWN, because the per-kernel cudaEvent syncs
 * needed to measure them serialize the pipeline and inflate the total.  The
 * headline number always comes from a clean build.
 * ---------------------------------------------------------------------------*/
typedef struct {
    int    iterations;            /* loops until delta <= threshold           */
    double clustering_sec;        /* whole convergence loop, wall clock       */
    double transfer_sec;          /* H2D + D2H performed inside the loop      */

    /* zero unless BENCH_KERNEL_BREAKDOWN */
    float  find_nearest_ms;       /* accumulated over all iterations          */
    float  reduce_coord_ms;
    float  reduce_changed_ms;

    /* launch geometry, so the reported numbers are self-describing */
    int    num_blocks;
    int    threads_per_block;
    int    reduction_threads;
    size_t shared_bytes;

    /* device identity */
    char   gpu_name[256];
    int    sm_count;
    int    cc_major;
    int    cc_minor;
} kmeans_perf_t;

int cuda_kmeans(float**, int, int, int, float, int*, int*, float***,
                kmeans_perf_t*);
float** file_read(char*, int*, int*);
int     file_write(char*, int, int, int, float**, int*);
double  wtime(void);

int hello();
#endif

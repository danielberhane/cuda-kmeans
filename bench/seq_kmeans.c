/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*   File:         seq_kmeans.c                                              */
/*   Description:  Sequential CPU k-means baseline for the CUDA version.     */
/*                                                                           */
/*   This exists to produce an honest speedup number, which means it must    */
/*   perform the SAME computation as src/cuda_kmeans.cu:                     */
/*                                                                           */
/*     - centroids initialised to the first K data objects                   */
/*     - squared Euclidean distance accumulated in float, no sqrt            */
/*       (mirrors euclid_dist() in cuda_kmeans.cu)                           */
/*     - convergence when delta = changed/numObjs <= threshold               */
/*     - hard cap of 500 iterations                                          */
/*                                                                           */
/*   It must converge in the same number of iterations as the CUDA build.    */
/*   If it does not, the two are not doing the same work and any speedup     */
/*   computed from them is meaningless -- see bench/run_bench.slurm.         */
/*                                                                           */
/*   Deliberately NOT handicapped: built at -O3, uses the cache-friendly     */
/*   AoS layout rather than the SoA layout the GPU needs, and can use all    */
/*   cores via OpenMP. Comparing a GPU against an artificially slow CPU      */
/*   baseline inflates speedup, so the sweep reports both the 1-thread       */
/*   and the all-core figure.                                                */
/*                                                                           */
/*   Author: Daniel Berhane Araya -- MIT License, see LICENSE                */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <sys/time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define MAX_ITERATIONS 500

static double wtime(void)
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec / 1000000.0;
}

/* Reads the same format as src/cuda_io.cu:file_read() -- one object per line,
   first whitespace-separated token is an id and is discarded, the remaining
   tokens are coordinates. numCoords is inferred from the first object.
   Returns a flat [numObjs * numCoords] array in AoS order. */
static float *read_objects(const char *filename, int *numObjs, int *numCoords)
{
    FILE   *f = fopen(filename, "r");
    char   *line = NULL;
    size_t  cap = 0;
    ssize_t len;
    int     n = 0, d = 0, i;
    float  *data = NULL;
    size_t  alloc = 0;

    if (f == NULL) {
        fprintf(stderr, "Error: no such file (%s)\n", filename);
        return NULL;
    }

    while ((len = getline(&line, &cap, f)) != -1) {
        char *p = line, *end;

        /* First field is an object id and is discarded, matching file_read(). */
        strtod(p, &end);
        if (end == p) continue;              /* blank or non-numeric line */
        p = end;

        if (d == 0) {                        /* first object fixes the width */
            char *q = p;
            for (;;) {
                strtod(q, &end);
                if (end == q) break;
                q = end;
                d++;
            }
            if (d == 0) {
                fprintf(stderr, "Error: no coordinates in %s\n", filename);
                fclose(f); free(line); return NULL;
            }
        }

        if ((size_t)(n + 1) * d > alloc) {
            alloc = alloc ? alloc * 2 : (size_t)d * 4096;
            data = (float *) realloc(data, alloc * sizeof(float));
            assert(data != NULL);
        }

        for (i = 0; i < d; i++) {
            data[(size_t)n * d + i] = (float) strtod(p, &end);
            if (end == p) {
                fprintf(stderr, "Error: object %d has fewer than %d coordinates\n", n, d);
                fclose(f); free(line); free(data); return NULL;
            }
            p = end;
        }
        n++;
    }
    (void) len;

    fclose(f);
    free(line);
    *numObjs   = n;
    *numCoords = d;
    return data;
}

int main(int argc, char **argv)
{
    int    numClusters, numCoords = 0, numObjs = 0;
    float  threshold;
    const char *filename;
    int    emit_csv = 0;
    int    i, j, k, loop = 0;
    float  delta = 1.0f;
    int    nthreads = 1;

    if (argc < 6) {
        printf("Usage: %s <num_clusters> <num_dimensions> <num_points> "
               "<threshold> <input_file> [--csv]\n", argv[0]);
        printf("  (num_dimensions and num_points are accepted for symmetry with\n"
               "   the CUDA driver; both are re-derived from the input file)\n");
        return 1;
    }

    numClusters = atoi(argv[1]);
    threshold   = (float) atof(argv[4]);
    filename    = argv[5];
    if (argc > 6 && strcmp(argv[6], "--csv") == 0) emit_csv = 1;

#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif

    float *objects = read_objects(filename, &numObjs, &numCoords);
    if (objects == NULL) return 1;

    if (numClusters > numObjs) {
        fprintf(stderr, "Error: numClusters (%d) > numObjs (%d)\n", numClusters, numObjs);
        return 1;
    }

    /* Same initialisation as cuda_kmeans(): centroid k is object k. */
    float *clusters = (float *) malloc((size_t)numClusters * numCoords * sizeof(float));
    assert(clusters != NULL);
    for (k = 0; k < numClusters; k++)
        for (j = 0; j < numCoords; j++)
            clusters[(size_t)k * numCoords + j] = objects[(size_t)k * numCoords + j];

    int *membership = (int *) malloc((size_t)numObjs * sizeof(int));
    assert(membership != NULL);
    for (i = 0; i < numObjs; i++) membership[i] = -1;

    double *sum = (double *) malloc((size_t)numClusters * numCoords * sizeof(double));
    long   *cnt = (long *)   malloc((size_t)numClusters * sizeof(long));
    assert(sum != NULL && cnt != NULL);

    const double t_start = wtime();

    do {
        long changed = 0;
        loop++;

        memset(sum, 0, (size_t)numClusters * numCoords * sizeof(double));
        memset(cnt, 0, (size_t)numClusters * sizeof(long));

        /* Without -fopenmp the pragmas are ignored and this region simply runs
           once, serially, which is exactly the 1-thread baseline. */
        #pragma omp parallel
        {
            double *l_sum = (double *) calloc((size_t)numClusters * numCoords, sizeof(double));
            long   *l_cnt = (long *)   calloc((size_t)numClusters, sizeof(long));
            long    l_changed = 0;
            int     ii, jj, kk;

            #pragma omp for schedule(static)
            for (ii = 0; ii < numObjs; ii++) {
                float best = -1.0f;
                int   bestk = 0;
                const float *obj = &objects[(size_t)ii * numCoords];

                for (kk = 0; kk < numClusters; kk++) {
                    /* float accumulator, squared distance, no sqrt --
                       matches euclid_dist() in cuda_kmeans.cu */
                    float dist = 0.0f;
                    const float *cen = &clusters[(size_t)kk * numCoords];
                    for (jj = 0; jj < numCoords; jj++) {
                        float t = obj[jj] - cen[jj];
                        dist += t * t;
                    }
                    if (best < 0.0f || dist < best) { best = dist; bestk = kk; }
                }

                if (membership[ii] != bestk) { l_changed++; membership[ii] = bestk; }

                l_cnt[bestk]++;
                for (jj = 0; jj < numCoords; jj++)
                    l_sum[(size_t)bestk * numCoords + jj] += obj[jj];
            }

            #pragma omp critical
            {
                changed += l_changed;
                for (kk = 0; kk < numClusters; kk++) {
                    cnt[kk] += l_cnt[kk];
                    for (jj = 0; jj < numCoords; jj++)
                        sum[(size_t)kk * numCoords + jj] += l_sum[(size_t)kk * numCoords + jj];
                }
            }

            free(l_sum);
            free(l_cnt);
        }

        for (k = 0; k < numClusters; k++) {
            if (cnt[k] == 0) {
                /* cuda_kmeans.cu divides by the count unconditionally and would
                   produce NaN here. Holding the centroid keeps this baseline
                   well-defined; it does not arise for the bundled datasets. */
                fprintf(stderr, "warning: cluster %d empty at iteration %d\n", k, loop);
                continue;
            }
            for (j = 0; j < numCoords; j++)
                clusters[(size_t)k * numCoords + j] =
                    (float)(sum[(size_t)k * numCoords + j] / (double)cnt[k]);
        }

        delta = (float)changed / (float)numObjs;

    } while ((delta > threshold) && (loop < MAX_ITERATIONS));

    const double t_elapsed = wtime() - t_start;

    if (emit_csv) {
        /* Column order must match bench/run_bench.slurm's header line. */
        printf("%s,%d,%d,%d,%g,%d,%.6f,%d\n",
               filename, numObjs, numCoords, numClusters, threshold,
               loop, t_elapsed, nthreads);
    } else {
        printf("\n*** Kmeans (sequential CPU baseline) ***\n\n");
        printf("Input file:       %s\n", filename);
        printf("numObjs         = %d\n", numObjs);
        printf("numCoords       = %d\n", numCoords);
        printf("numClusters     = %d\n", numClusters);
        printf("threshold       = %g\n", threshold);
        printf("iterations      = %d\n", loop);
        printf("threads         = %d\n", nthreads);
        printf("Clustering time = %10.4f sec\n\n", t_elapsed);

        /* Written so the centroids can be diffed against the CUDA run. */
        char out[1024];
        snprintf(out, sizeof(out), "%s.seq_centres", filename);
        FILE *fp = fopen(out, "w");
        if (fp) {
            for (k = 0; k < numClusters; k++) {
                fprintf(fp, "%d", k);
                for (j = 0; j < numCoords; j++)
                    fprintf(fp, " %f", clusters[(size_t)k * numCoords + j]);
                fprintf(fp, "\n");
            }
            fclose(fp);
            printf("centroids -> %s\n\n", out);
        }
    }

    free(objects); free(clusters); free(membership); free(sum); free(cnt);
    return 0;
}

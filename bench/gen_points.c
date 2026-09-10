/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*   File:         gen_points.c                                              */
/*   Description:  Generates uniformly-distributed point files for the       */
/*                 scaling study.                                            */
/*                                                                           */
/*   The bundled datasets in data/ are ~100k points, which is too small to    */
/*   saturate a modern GPU: at that size the convergence loop is dominated   */
/*   by per-iteration launch and transfer latency rather than by arithmetic. */
/*   A speedup number measured only at 100k would understate the kernels.    */
/*   This generator produces the same uniform distribution at arbitrary N so */
/*   the scaling curve is a single consistent series.                        */
/*                                                                           */
/*   Uses xorshift64* rather than rand() so output is bit-identical across   */
/*   platforms and libc versions -- the generated files are reproducible     */
/*   from (seed, N, D) alone and therefore need not be committed.            */
/*                                                                           */
/*   Author: Daniel Berhane Araya -- MIT License, see LICENSE                */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

static uint64_t rng_state;

static double next_uniform(void)      /* [0,1) */
{
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return (double)((rng_state * 2685821657736338717ULL) >> 11) / 9007199254740992.0;
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <num_points> <num_dims> <out_file> [seed]\n", argv[0]);
        fprintf(stderr, "  Matches the format of data/points_*.txt:\n");
        fprintf(stderr, "  one object per line, 1-based id then <num_dims> coords in [0,100).\n");
        return 1;
    }

    long        n    = atol(argv[1]);
    int         d    = atoi(argv[2]);
    const char *out  = argv[3];
    rng_state        = (argc > 4) ? (uint64_t) atoll(argv[4]) : 88172645463325252ULL;

    if (n <= 0 || d <= 0) { fprintf(stderr, "Error: num_points and num_dims must be positive\n"); return 1; }
    if (rng_state == 0)   { fprintf(stderr, "Error: seed must be non-zero (xorshift)\n"); return 1; }

    FILE *f = fopen(out, "w");
    if (f == NULL) { fprintf(stderr, "Error: cannot write %s\n", out); return 1; }

    /* A large buffer matters here: at N=10M this loop is the bottleneck. */
    static char buf[1 << 20];
    setvbuf(f, buf, _IOFBF, sizeof(buf));

    for (long i = 0; i < n; i++) {
        fprintf(f, "%ld", i + 1);
        for (int j = 0; j < d; j++)
            fprintf(f, " %.9f", next_uniform() * 100.0);
        fputc('\n', f);
    }

    fclose(f);
    fprintf(stderr, "wrote %ld points x %d dims -> %s\n", n, d, out);
    return 0;
}

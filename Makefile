# Originally based on code by Wei-keng Liao and Serban Giuroiu (MIT License)
# Modified for multi-dimensional CUDA k-means clustering

# ------------------------------------------------------------------------------

.KEEP_STATE:

all: cuda_kmeans

SRCDIR      = src
BENCHDIR    = bench
BUILDDIR    = build

DFLAGS      =
OPTFLAGS    = -O2
INCFLAGS    = -I$(SRCDIR)
CFLAGS      = $(OPTFLAGS) $(DFLAGS) $(INCFLAGS) -DBLOCK_SHARED_MEM_OPTIMIZATION=1
NVCCFLAGS   = $(CFLAGS)
LDFLAGS     = $(OPTFLAGS)

NVCC        = nvcc
CC         ?= cc

# Host-side benchmark tools are plain C and need no CUDA toolchain, so they
# build anywhere. -O3 because a handicapped CPU baseline would inflate the
# reported speedup.
HOSTFLAGS   = -O3 -Wall -Wextra

# ------------------------------------------------------------------------------

SOURCES     = $(SRCDIR)/cuda_main.cu $(SRCDIR)/cuda_io.cu $(SRCDIR)/cuda_wtime.cu $(SRCDIR)/cuda_kmeans.cu
OBJECTS     = $(patsubst $(SRCDIR)/%.cu,$(BUILDDIR)/%.o,$(SOURCES))

# Separate object dir for the instrumented build so the two never share stale
# objects compiled with different -D flags.
BENCHOBJS   = $(patsubst $(SRCDIR)/%.cu,$(BUILDDIR)/bench_%.o,$(SOURCES))

$(BUILDDIR)/%.o : $(SRCDIR)/%.cu | $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<

$(BUILDDIR)/bench_%.o : $(SRCDIR)/%.cu | $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) -DBENCH_KERNEL_BREAKDOWN -o $@ -c $<

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

cuda_kmeans: $(OBJECTS)
	$(NVCC) $(LDFLAGS) -o $@ $(OBJECTS)

# Per-kernel cudaEvent timings. The syncs these require serialize the pipeline,
# so this build reports a HIGHER total than ./cuda_kmeans -- use it for the
# breakdown only, never for the headline number.
cuda_kmeans_bench: $(BENCHOBJS)
	$(NVCC) $(LDFLAGS) -o $@ $(BENCHOBJS)

# ------------------------------------------------------------------------------
# Host-only benchmark tools

bench-tools: seq_kmeans seq_kmeans_omp gen_points

seq_kmeans: $(BENCHDIR)/seq_kmeans.c
	$(CC) $(HOSTFLAGS) -o $@ $<

# Apple clang ships without OpenMP, so probe before using it rather than
# failing the whole bench-tools target on a dev machine. run_bench.slurm
# treats a missing binary as "no multicore baseline available".
seq_kmeans_omp: $(BENCHDIR)/seq_kmeans.c
	@if echo 'int main(void){return 0;}' | $(CC) -fopenmp -x c - -o /dev/null >/dev/null 2>&1; then \
	    echo "$(CC) $(HOSTFLAGS) -fopenmp -o $@ $<"; \
	    $(CC) $(HOSTFLAGS) -fopenmp -o $@ $< ; \
	else \
	    echo "note: $(CC) has no OpenMP support -- skipping $@ (multicore CPU baseline unavailable)"; \
	fi

gen_points: $(BENCHDIR)/gen_points.c
	$(CC) $(HOSTFLAGS) -o $@ $<

# ------------------------------------------------------------------------------
clean:
	rm -rf $(BUILDDIR) cuda_kmeans cuda_kmeans_bench seq_kmeans seq_kmeans_omp gen_points

.PHONY: all clean bench-tools

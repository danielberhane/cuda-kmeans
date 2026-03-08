# Originally based on code by Wei-keng Liao and Serban Giuroiu (MIT License)
# Modified for multi-dimensional CUDA k-means clustering

# ------------------------------------------------------------------------------

.KEEP_STATE:

all: cuda_kmeans

SRCDIR      = src
BUILDDIR    = build

DFLAGS      =
OPTFLAGS    = -O2
INCFLAGS    = -I$(SRCDIR)
CFLAGS      = $(OPTFLAGS) $(DFLAGS) $(INCFLAGS) -DBLOCK_SHARED_MEM_OPTIMIZATION=1
NVCCFLAGS   = $(CFLAGS)
LDFLAGS     = $(OPTFLAGS)

NVCC        = nvcc

# ------------------------------------------------------------------------------

SOURCES     = $(SRCDIR)/cuda_main.cu $(SRCDIR)/cuda_io.cu $(SRCDIR)/cuda_wtime.cu $(SRCDIR)/cuda_kmeans.cu
OBJECTS     = $(patsubst $(SRCDIR)/%.cu,$(BUILDDIR)/%.o,$(SOURCES))

$(BUILDDIR)/%.o : $(SRCDIR)/%.cu | $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

cuda_kmeans: $(OBJECTS)
	$(NVCC) $(LDFLAGS) -o $@ $(OBJECTS)

# ------------------------------------------------------------------------------
clean:
	rm -rf $(BUILDDIR) cuda_kmeans

.PHONY: all clean

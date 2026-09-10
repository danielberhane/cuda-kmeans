#!/bin/bash
# =============================================================================
#  run_bench.slurm -- benchmark sweep for cuda-kmeans
#
#  ,--------------------------------------------------------------------.
#  |  BEFORE SUBMITTING: fill in the four PLACEHOLDER values below.      |
#  |  They are intentionally left blank rather than guessed, because     |
#  |  partition names, GPU types and CUDA module strings differ between  |
#  |  clusters and change over time. Run these three commands on the     |
#  |  login node and use their output:                                   |
#  |                                                                     |
#  |    sinfo -s                  # partition names + availability       |
#  |    scontrol show partition   # which partitions expose GPUs (gres)  |
#  |    module avail cuda         # the exact CUDA module string         |
#  |                                                                     |
#  |  Then:  sbatch bench/run_bench.slurm                                |
#  `--------------------------------------------------------------------'
# =============================================================================

#SBATCH --job-name=cuda-kmeans-bench
#SBATCH --output=bench/results/slurm-%j.out
#SBATCH --error=bench/results/slurm-%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

## PLACEHOLDER 1 -- partition that has GPUs (from `sinfo -s`)
###SBATCH --partition=PLACEHOLDER_PARTITION

## PLACEHOLDER 2 -- GPU request (from `scontrol show partition`); often
##                  --gres=gpu:1  or  --gres=gpu:a100:1
###SBATCH --gres=gpu:1

## PLACEHOLDER 3 -- cores for the multicore CPU baseline. Set to a full
##                  socket so the OpenMP comparison is meaningful.
###SBATCH --cpus-per-task=16

set -euo pipefail

## PLACEHOLDER 4 -- exact module string(s) from `module avail cuda`
# module load PLACEHOLDER_CUDA_MODULE
# module load gcc            # if the default gcc is too old for nvcc

# -----------------------------------------------------------------------------
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p bench/results bench/data

REPEATS=3                       # runs per configuration; the MINIMUM is kept,
                                # which is the standard choice for benchmark
                                # timing because noise is one-sided (it only
                                # ever makes a run slower, never faster)
STAMP="$(date +%Y%m%d-%H%M%S)"
GPU_CSV="bench/results/gpu-${STAMP}.csv"
CPU_CSV="bench/results/cpu-${STAMP}.csv"
LOG="bench/results/run-${STAMP}.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "=== environment ==="
{ nvidia-smi || true; nvcc --version || true; gcc --version | head -1 || true; } >> "$LOG" 2>&1

# -----------------------------------------------------------------------------
log "=== build ==="
make clean
make GENCODE=-arch=native cuda_kmeans        # clean build -- headline number
make GENCODE=-arch=native cuda_kmeans_bench  # instrumented -- breakdown only
make bench-tools            # seq_kmeans, seq_kmeans_omp, gen_points

HAVE_OMP=0
[ -x ./seq_kmeans_omp ] && HAVE_OMP=1
log "multicore CPU baseline available: $HAVE_OMP"

# -----------------------------------------------------------------------------
# Scaling datasets.
#
# The bundled files are ~100k points. At that size the convergence loop is
# dominated by per-iteration launch and transfer latency, not arithmetic, so a
# speedup measured only there understates the kernels. These larger sizes let
# the curve show where the GPU actually pulls ahead.
#
# Regenerated rather than committed: gen_points is deterministic given
# (seed, N, D), so these files are reproducible and stay out of git.
# -----------------------------------------------------------------------------
SCALING_SIZES=(100000 250000 500000 1000000 2000000 5000000)

log "=== generating scaling datasets ==="
for n in "${SCALING_SIZES[@]}"; do
    f="bench/data/scale_${n}_3d.txt"
    if [ ! -s "$f" ]; then
        ./gen_points "$n" 3 "$f" 88172645463325252
    fi
done

# -----------------------------------------------------------------------------
# CSV headers -- column order must match the printf() in src/cuda_main.cu and
# bench/seq_kmeans.c respectively.
# -----------------------------------------------------------------------------
echo "file,N,D,K,threshold,iterations,loop_sec,transfer_sec,endtoend_sec,io_sec,find_ms,coord_ms,changed_ms,blocks,threads_per_block,reduction_threads,shared_bytes,gpu_name,sm_count,cc" > "$GPU_CSV"
echo "file,N,D,K,threshold,iterations,elapsed_sec,threads" > "$CPU_CSV"

# Take the fastest of REPEATS runs. Field 7 is the timing column in both CSVs.
best_of() {
    local out="" best="" line t
    for _ in $(seq "$REPEATS"); do
        line="$("$@")"
        t="$(echo "$line" | cut -d, -f7)"
        if [ -z "$best" ] || awk "BEGIN{exit !($t < $best)}"; then best="$t"; out="$line"; fi
    done
    echo "$out"
}

run_config() {
    local file="$1" dims="$2" k="$3" thresh="${4:-0.001}"
    local n; n="$(wc -l < "$file")"

    log "  ${file##*/}  N=$n D=$dims K=$k"

    # GPU context creation costs ~100-300 ms on first touch; without this
    # warm-up the first measured run of the job absorbs it.
    ./cuda_kmeans "$k" "$dims" "$n" "$thresh" "$file" --csv > /dev/null

    best_of ./cuda_kmeans "$k" "$dims" "$n" "$thresh" "$file" --csv >> "$GPU_CSV"
    best_of ./seq_kmeans  "$k" "$dims" "$n" "$thresh" "$file" --csv >> "$CPU_CSV"
    if [ "$HAVE_OMP" = "1" ]; then
        OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" \
            best_of ./seq_kmeans_omp "$k" "$dims" "$n" "$thresh" "$file" --csv >> "$CPU_CSV"
    fi
}

# -----------------------------------------------------------------------------
log "=== bundled datasets: dimensionality x K ==="
for k in 4 8 16; do
    run_config data/points_1d.txt  1  "$k"
    run_config data/points_3d.txt  3  "$k"
    run_config data/points_10d.txt 10 "$k"
done

log "=== scaling study: N sweep at D=3, K=4 ==="
for n in "${SCALING_SIZES[@]}"; do
    run_config "bench/data/scale_${n}_3d.txt" 3 4
done

log "=== per-kernel breakdown (instrumented build; syncs inflate the total) ==="
BREAKDOWN="bench/results/breakdown-${STAMP}.txt"
for n in 100000 1000000 5000000; do
    f="bench/data/scale_${n}_3d.txt"
    { echo "--- N=$n D=3 K=4 ---"; ./cuda_kmeans_bench 4 3 "$n" 0.001 "$f"; } >> "$BREAKDOWN"
done

# -----------------------------------------------------------------------------
# Correctness gate. The CPU baseline and the CUDA build must converge in the
# same number of iterations; if they do not, they are not performing the same
# computation and any speedup derived from them is meaningless.
# -----------------------------------------------------------------------------
log "=== verifying CPU and GPU agree on iteration count ==="
MISMATCH=0
while IFS=, read -r gfile gn gd gk gthr gitr _rest; do
    [ "$gfile" = "file" ] && continue
    citr="$(awk -F, -v f="$gfile" -v k="$gk" '$1==f && $4==k {print $6; exit}' "$CPU_CSV")"
    if [ -n "$citr" ] && [ "$citr" != "$gitr" ]; then
        log "  MISMATCH  $gfile K=$gk : gpu=$gitr cpu=$citr"
        MISMATCH=1
    fi
done < "$GPU_CSV"

if [ "$MISMATCH" = "1" ]; then
    log "!! Iteration counts diverge -- do NOT publish speedups from this run."
else
    log "OK: CPU and GPU converge identically on every configuration."
fi

log "=== done ==="
log "  GPU results : $GPU_CSV"
log "  CPU results : $CPU_CSV"
log "  breakdown   : $BREAKDOWN"
log ""
log "Paste $GPU_CSV and $CPU_CSV back to build the README table and scaling chart."

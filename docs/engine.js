// k-means matching src/cuda_kmeans.cu semantics exactly:
//   - centroids initialised to the first K data objects
//   - squared Euclidean distance in a float accumulator, no sqrt
//   - convergence when delta = changed/numObjs <= threshold
//   - hard cap of 500 iterations
//
// Also records the per-block quantities the CUDA version computes, so anything
// visualised from this trace reflects real work rather than decorative motion.
//
// Pure: no I/O, no DOM. Runs unchanged in the browser, in a worker, and under
// Node (tools/kmeans-engine.mjs adds file reading; tools/verify-trace.mjs gates
// it against values measured from the real CUDA build).

export const MAX_ITERATIONS = 500;

export function nextPowerOfTwo(n) {          // mirrors nextPowerOfTwo() in the .cu
  n--; n |= n >> 1; n |= n >> 2; n |= n >> 4; n |= n >> 8; n |= n >> 16;
  return n + 1;
}

export function sharedLayout(K, D, blockSize) {
  // Declaration order from cuda_kmeans.cu:72-77.
  const regions = [
    { name: 's_objects',      bytes: blockSize * D * 4, note: 'point tile' },
    { name: 's_clusters',     bytes: K * D * 4,         note: 'centroids' },
    { name: 's_sum_clusters', bytes: K * 4,             note: 'counts' },
    { name: 's_sum_coords',   bytes: K * D * 4,         note: 'coord sums' },
    { name: 's_memb_changed', bytes: blockSize,         note: 'changed flags' },
    { name: 's_membership',   bytes: blockSize,         note: 'assignment' },
  ];
  return { regions, total: regions.reduce((a, r) => a + r.bytes, 0) };
}

export function runKMeans({ X, N, D }, {
  K = 4, threshold = 0.001, blockSize = 128, maxIter = MAX_ITERATIONS,
  keepMembership = true,
} = {}) {
  const numBlocks = Math.ceil(N / blockSize);
  const C = new Float32Array(K * D);
  for (let k = 0; k < K; k++)
    for (let j = 0; j < D; j++) C[k * D + j] = X[k * D + j];

  let prev = new Int32Array(N).fill(-1);
  const trace = [];

  for (let iter = 1; iter <= maxIter; iter++) {
    const m = new Uint8Array(N);
    const blockChanged = new Uint16Array(numBlocks);
    let changed = 0;

    for (let i = 0; i < N; i++) {
      let best = -1, bestk = 0;
      const base = i * D;
      for (let k = 0; k < K; k++) {
        let dist = 0;
        const cb = k * D;
        for (let j = 0; j < D; j++) {
          const t = X[base + j] - C[cb + j];
          dist += t * t;
        }
        if (best < 0 || dist < best) { best = dist; bestk = k; }
      }
      m[i] = bestk;
      if (bestk !== prev[i]) { changed++; blockChanged[(i / blockSize) | 0]++; }
    }
    prev = Int32Array.from(m);

    const cnt = new Float64Array(K), sum = new Float64Array(K * D);
    for (let i = 0; i < N; i++) {
      cnt[m[i]]++;
      const base = i * D, cb = m[i] * D;
      for (let j = 0; j < D; j++) sum[cb + j] += X[base + j];
    }
    for (let k = 0; k < K; k++)
      if (cnt[k]) for (let j = 0; j < D; j++) C[k * D + j] = sum[k * D + j] / cnt[k];

    const delta = changed / N;
    trace.push({
      iter, changed, delta,
      blockChanged,
      counts: Array.from(cnt, Number),
      centroids: Float32Array.from(C),
      membership: keepMembership ? m : null,
    });
    if (delta <= threshold) break;
  }

  const shared = sharedLayout(K, D, blockSize);
  return {
    trace, N, D, K, threshold, blockSize, numBlocks,
    reductionThreads: nextPowerOfTwo(numBlocks),
    shared: shared.regions,
    sharedBytes: shared.total,
  };
}

// A synthetic clustered set, for contrast with the bundled benchmark data
// (which is uniformly random and therefore has no structure to recover).
// Deterministic given the seed, via xorshift64* -- same generator as
// bench/gen_points.c.
export function makeClustered(N, D, K, seed = 0x2545F491n) {
  let s = BigInt(seed) || 1n;
  const M = (1n << 64n) - 1n;
  const rnd = () => {
    s ^= (s >> 12n) & M; s = (s ^ ((s << 25n) & M)) & M; s ^= (s >> 27n) & M;
    return Number(((s * 2685821657736338717n) & M) >> 11n) / 9007199254740992;
  };
  const gauss = () => {
    const u = Math.max(rnd(), 1e-12), v = rnd();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  const centres = [];
  for (let k = 0; k < K; k++) {
    const c = [];
    for (let j = 0; j < D; j++) c.push(12 + rnd() * 76);
    centres.push(c);
  }
  const X = new Float32Array(N * D);
  for (let i = 0; i < N; i++) {
    const c = centres[i % K];
    for (let j = 0; j < D; j++) X[i * D + j] = Math.min(100, Math.max(0, c[j] + gauss() * 6.5));
  }
  return { X, N, D };
}

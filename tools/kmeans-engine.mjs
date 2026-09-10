// Deterministic k-means matching src/cuda_kmeans.cu semantics exactly:
// first-K initialisation, squared float distance without sqrt, delta =
// changed/N convergence, 500-iteration cap. Also records the per-block
// quantities the CUDA version computes, so a visualisation built on this
// trace depicts real work rather than decorative motion.

import fs from 'node:fs';

export function readPoints(file) {
  const lines = fs.readFileSync(file, 'utf8').trim().split('\n');
  const N = lines.length;
  const D = lines[0].trim().split(/\s+/).length - 1;   // first token is an id
  const X = new Float32Array(N * D);
  for (let i = 0; i < N; i++) {
    const p = lines[i].trim().split(/\s+/);
    for (let j = 0; j < D; j++) X[i * D + j] = +p[j + 1];
  }
  return { X, N, D };
}

export function runKMeans({ X, N, D }, K, threshold = 0.001, blockSize = 128, maxIter = 500) {
  const numBlocks = Math.ceil(N / blockSize);
  const C = new Float32Array(K * D);
  for (let k = 0; k < K; k++)                          // centroid k = object k
    for (let j = 0; j < D; j++) C[k * D + j] = X[k * D + j];

  let prev = new Int32Array(N).fill(-1);
  const trace = [];

  for (let iter = 1; iter <= maxIter; iter++) {
    const m = new Int32Array(N);
    const blockChanged = new Uint16Array(numBlocks);
    let changed = 0;

    for (let i = 0; i < N; i++) {
      let best = -1, bestk = 0;
      for (let k = 0; k < K; k++) {
        let dist = 0;                                   // float accumulator, no sqrt
        for (let j = 0; j < D; j++) {
          const t = X[i * D + j] - C[k * D + j];
          dist += t * t;
        }
        if (best < 0 || dist < best) { best = dist; bestk = k; }
      }
      m[i] = bestk;
      if (bestk !== prev[i]) { changed++; blockChanged[(i / blockSize) | 0]++; }
    }
    prev = m;

    const cnt = new Float64Array(K), sum = new Float64Array(K * D);
    for (let i = 0; i < N; i++) {
      cnt[m[i]]++;
      for (let j = 0; j < D; j++) sum[m[i] * D + j] += X[i * D + j];
    }
    for (let k = 0; k < K; k++)
      if (cnt[k]) for (let j = 0; j < D; j++) C[k * D + j] = sum[k * D + j] / cnt[k];

    const delta = changed / N;
    trace.push({
      iter, changed, delta, blockChanged,
      counts: Array.from(cnt, Number),
      centroids: Float32Array.from(C),
    });
    if (delta <= threshold) break;
  }

  // Shared memory layout, in declaration order from cuda_kmeans.cu:72-77
  const shared = [
    { name: 's_objects',      bytes: blockSize * D * 4, label: 'point tile' },
    { name: 's_clusters',     bytes: K * D * 4,         label: 'centroids' },
    { name: 's_sum_clusters', bytes: K * 4,             label: 'counts' },
    { name: 's_sum_coords',   bytes: K * D * 4,         label: 'coord sums' },
    { name: 's_memb_changed', bytes: blockSize,         label: 'changed' },
    { name: 's_membership',   bytes: blockSize,         label: 'assignment' },
  ];

  return {
    trace, N, D, K, blockSize, numBlocks,
    reductionThreads: nextPow2(numBlocks),
    shared,
    sharedBytes: shared.reduce((a, r) => a + r.bytes, 0),
  };
}

function nextPow2(n) {                                  // mirrors nextPowerOfTwo()
  n--; n |= n >> 1; n |= n >> 2; n |= n >> 4; n |= n >> 8; n |= n >> 16;
  return n + 1;
}

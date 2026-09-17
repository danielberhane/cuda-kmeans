// Traces run off the main thread: at N=99,968 and K=16 a full run is a second
// or more of solid arithmetic, and doing that inline would freeze the UI on
// every slider change.

import { runKMeans, makeClustered } from './engine.js';

let benchmark = null;   // the real dataset, loaded once

self.onmessage = async (e) => {
  const { type, payload } = e.data;

  // docs/data/points_3d.txt is a copy of data/points_3d.txt in the same
  // "id x y z" text format the CUDA build reads, so the demo runs on exactly
  // the input the benchmark did. Parsing 100k lines takes tens of
  // milliseconds in a worker; a packed binary would save little and a
  // .bin in the tree is what GitHub's spam scan objected to.
  if (type === 'load') {
    const res = await fetch(payload.url);
    const lines = (await res.text()).trim().split('\n');
    const N = lines.length;
    const D = lines[0].trim().split(/\s+/).length - 1;   // first token is an id
    const X = new Float32Array(N * D);
    for (let i = 0; i < N; i++) {
      const p = lines[i].trim().split(/\s+/);
      for (let j = 0; j < D; j++) X[i * D + j] = +p[j + 1];
    }
    benchmark = { X, N, D };
    self.postMessage({ type: 'loaded', payload: { N, D } });
    return;
  }

  if (type === 'run') {
    const { dataset, K, threshold, blockSize } = payload;
    const pts = dataset === 'clustered'
      ? makeClustered(benchmark.N, benchmark.D, K)
      : benchmark;

    const t0 = performance.now();
    const r = runKMeans(pts, { K, threshold, blockSize });
    const ms = performance.now() - t0;

    // Structured clone would copy several MB per run; transfer instead.
    const transfer = [];
    for (const t of r.trace) {
      transfer.push(t.blockChanged.buffer, t.centroids.buffer);
      if (t.membership) transfer.push(t.membership.buffer);
    }
    const coords = pts.X.slice();          // the view the renderer plots
    transfer.push(coords.buffer);

    self.postMessage({ type: 'result', payload: { ...r, coords, ms } }, transfer);
  }
};

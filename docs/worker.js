// Traces run off the main thread: at N=99,968 and K=16 a full run is a second
// or more of solid arithmetic, and doing that inline would freeze the UI on
// every slider change.

import { runKMeans, makeClustered } from './engine.js';

let benchmark = null;   // the real dataset, loaded once

self.onmessage = async (e) => {
  const { type, payload } = e.data;

  if (type === 'load') {
    const res = await fetch(payload.url);
    const buf = await res.arrayBuffer();
    const N = new Int32Array(buf, 0, 1)[0];
    const D = new Int32Array(buf, 4, 1)[0];
    benchmark = { X: new Float32Array(buf, 8, N * D), N, D };
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

// Node-side wrapper: file reading plus the shared browser engine, so the
// diagram generator, the verification gate and the interactive demo all run
// the exact same implementation and cannot drift apart.

import fs from 'node:fs';
export * from '../docs/engine.js';
import { runKMeans as run } from '../docs/engine.js';

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

// Keeps the positional call signature the existing tools use.
export function runKMeans(pts, K = 4, threshold = 0.001, blockSize = 128) {
  return run(pts, { K, threshold, blockSize });
}

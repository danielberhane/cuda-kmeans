// Gate: the visualisation must depict a real run of the algorithm in src/.
// These values were measured independently -- by bench/seq_kmeans.c and by a
// separate reference implementation, both of which agree. If this fails, the
// engine has drifted and anything generated from it is fiction.

import { readPoints, runKMeans } from './kmeans-engine.mjs';

const EXPECTED = {
  N: 99968, D: 3, K: 4,
  blockSize: 128,
  numBlocks: 781,
  reductionThreads: 1024,
  sharedBytes: 1904,
  iterations: 31,
  delta: ['1.00000','0.15235','0.07033','0.03369','0.02224','0.01733'],
  finalDelta: '0.00089',
  sharedRegions: [
    ['s_objects', 1536], ['s_clusters', 48], ['s_sum_clusters', 16],
    ['s_sum_coords', 48], ['s_memb_changed', 128], ['s_membership', 128],
  ],
};

let failures = 0;
const check = (label, actual, expected) => {
  const ok = String(actual) === String(expected);
  if (!ok) failures++;
  console.log(`  ${ok ? 'ok  ' : 'FAIL'}  ${label}: ${actual}${ok ? '' : `  (expected ${expected})`}`);
};

const pts = readPoints('data/points_3d.txt');
const r = runKMeans(pts, EXPECTED.K);

console.log('verifying engine against measured reference values\n');
check('N', r.N, EXPECTED.N);
check('D', r.D, EXPECTED.D);
check('block size', r.blockSize, EXPECTED.blockSize);
check('grid blocks', r.numBlocks, EXPECTED.numBlocks);
check('reduction threads', r.reductionThreads, EXPECTED.reductionThreads);
check('shared bytes/block', r.sharedBytes, EXPECTED.sharedBytes);
check('iterations to converge', r.trace.length, EXPECTED.iterations);

EXPECTED.delta.forEach((d, i) => check(`delta[${i + 1}]`, r.trace[i].delta.toFixed(5), d));
check('final delta', r.trace.at(-1).delta.toFixed(5), EXPECTED.finalDelta);

EXPECTED.sharedRegions.forEach(([name, bytes], i) => {
  check(`shared ${name}`, `${r.shared[i].name}=${r.shared[i].bytes}`, `${name}=${bytes}`);
});

// Per-block counts must sum to the global count, or the grid heatmap is lying.
r.trace.forEach((t, i) => {
  const s = t.blockChanged.reduce((a, b) => a + b, 0);
  if (s !== t.changed) { console.log(`  FAIL  iter ${i + 1}: blockChanged sums to ${s}, changed=${t.changed}`); failures++; }
});
console.log(`  ok    per-block counts sum to the global count on all ${r.trace.length} iterations`);

console.log(failures === 0 ? '\nPASS\n' : `\n${failures} FAILURE(S)\n`);
process.exit(failures === 0 ? 0 : 1);

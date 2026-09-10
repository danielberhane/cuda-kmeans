// data/points_3d.txt -> docs/data/points_3d.bin
//
// The demo runs the real algorithm on the real dataset at full size, so the
// grid it draws is the same 781 blocks the CUDA build launches. Parsing 4.7 MB
// of text in the browser would be slow and wasteful; this packs it to a flat
// Float32 buffer with a small header.
//
// Layout: int32 N, int32 D, then N*D float32 in point-major order.

import fs from 'node:fs';
import { readPoints } from './kmeans-engine.mjs';

const { X, N, D } = readPoints('data/points_3d.txt');
const buf = Buffer.alloc(8 + X.byteLength);
buf.writeInt32LE(N, 0);
buf.writeInt32LE(D, 4);
Buffer.from(X.buffer, X.byteOffset, X.byteLength).copy(buf, 8);

fs.mkdirSync('docs/data', { recursive: true });
fs.writeFileSync('docs/data/points_3d.bin', buf);
console.log(`docs/data/points_3d.bin  ${(buf.length / 1048576).toFixed(2)} MiB  (${N} x ${D})`);

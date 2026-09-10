// Emits assets/pipeline.svg -- an animated diagram of the CUDA execution model.
//
// Everything animated is driven by a real run of the algorithm
// (tools/kmeans-engine.mjs, gated by tools/verify-trace.mjs): the grid heatmap
// is per-block counts of points that actually changed cluster, and the shared
// memory regions are drawn to their real byte sizes.
//
// SVG rather than GIF because the subject is geometric -- far smaller, crisp at
// any zoom, and able to carry its own dark-mode CSS.
//
// Animation is CSS keyframes, not SMIL. SMIL cannot animate `class`, and
// animating `fill` directly would have to bake in literal colours, which kills
// theming. Instead each cell's activity sequence becomes a CSS class whose
// keyframes reference theme variables -- so one rule serves every cell sharing
// that sequence, and both themes work from the same markup.

import fs from 'node:fs';
import { readPoints, runKMeans } from './kmeans-engine.mjs';

const W = 900, H = 616;
const CYCLE = 4;                        // seconds per displayed iteration
const FRAMES = [1, 3, 8, 31];
const LOOP = CYCLE * FRAMES.length;

const r = runKMeans(readPoints('data/points_3d.txt'), 4);
const frames = FRAMES.map(i => r.trace[i - 1]);
const fmt = n => n.toLocaleString('en-US');
const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

const COLS = 62, CELL = 10, GAP = 2, PITCH = CELL + GAP;
const ROWS = Math.ceil(r.numBlocks / COLS);
const GRID_W = COLS * PITCH - GAP, GRID_X = Math.round((W - GRID_W) / 2), GRID_Y = 200;

// 0 is a distinct "idle" level; sqrt lifts low activity so it stays visible
const level = v => (v === 0 ? 0 : Math.min(4, 1 + Math.floor(Math.sqrt(v / r.blockSize) * 4)));

// Group cells by their level sequence so each distinct sequence needs one rule.
const seqOf = b => frames.map(f => level(f.blockChanged[b])).join('');
const seqs = new Map();
const cells = [];
for (let b = 0; b < r.numBlocks; b++) {
  const s = seqOf(b);
  if (!seqs.has(s)) seqs.set(s, 0);
  seqs.set(s, seqs.get(s) + 1);
  cells.push(
    `<rect class="c s${s}" x="${GRID_X + (b % COLS) * PITCH}" y="${GRID_Y + Math.floor(b / COLS) * PITCH}" width="${CELL}" height="${CELL}"/>`
  );
}
const stops = FRAMES.map((_, i) => (i * 100 / FRAMES.length).toFixed(2));
const seqCSS = [...seqs.keys()].map(s => {
  const kf = s.split('').map((lv, i) => `${stops[i]}%{fill:var(--l${lv})}`).join('');
  return `@keyframes k${s}{${kf}}.s${s}{animation:k${s} ${LOOP}s step-end infinite}`;
}).join('\n  ');

// shared memory bar, regions to real scale
const SM_X = 60, SM_W = W - 120, SM_Y = 400, SM_H = 34;
let acc = 0;
const smRegions = r.shared.map(s => {
  const x = SM_X + (acc / r.sharedBytes) * SM_W, w = (s.bytes / r.sharedBytes) * SM_W;
  acc += s.bytes;
  return { ...s, x, w };
});

// kernel pipeline
const kernels = [
  { n: 'find_nearest_cluster',   g: `${fmt(r.numBlocks)} × ${r.blockSize}` },
  { n: 'reduce_coord_clusters',  g: `${fmt(r.numBlocks)} × ${r.blockSize}` },
  { n: 'reduce_cluster_changed', g: `1 × ${fmt(r.reductionThreads)}`, warn: true },
];
const KB_Y = 492, KB_H = 64, KB_W = 236, KB_GAP = 26;
const KB_X0 = Math.round((W - (KB_W * 3 + KB_GAP * 2)) / 2);
const tokenStops = [
  [0, 20], [32, KB_W - 20], [42, KB_W + KB_GAP + 20], [62, 2 * KB_W + KB_GAP - 20],
  [68, 2 * (KB_W + KB_GAP) + 20], [84, 3 * KB_W + 2 * KB_GAP - 20], [100, 3 * KB_W + 2 * KB_GAP - 20],
];
const tokenKF = tokenStops.map(([p, x]) => `${p}%{transform:translateX(${x - 20}px)}`).join('');

const frameText = (i, cls, x, y, anchor, body) =>
  `<text class="${cls} fr fr${i}" x="${x}" y="${y}" text-anchor="${anchor}">${body}</text>`;

const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-labelledby="ttl dsc">
<title id="ttl">CUDA k-means execution model</title>
<desc id="dsc">${fmt(r.N)} points in ${r.D} dimensions are transposed to a coordinate-major layout for coalesced access, tiled across ${fmt(r.numBlocks)} thread blocks of ${r.blockSize} threads, and processed by three kernels per iteration. The grid heatmap shows how many points in each block changed cluster; it goes quiet as the algorithm converges over ${r.trace.length} iterations. The third kernel launches a single block, leaving most of the device idle.</desc>
<style>
  :root{
    --bg:#fbfbfa;--panel:#ffffff;--ink:#16171a;--dim:#6a6d73;--line:#dcdcd6;
    --l0:#e7e7e1;--l1:#cfe0f2;--l2:#8fbde3;--l3:#4b8dc9;--l4:#1f5e9e;
    --warn:#c2410c;--warnbg:#fdf1e7;--flow:#1f5e9e;--sm:#8fbde3;
  }
  @media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
    --bg:#131519;--panel:#1b1e23;--ink:#e9e9e5;--dim:#9a9da4;--line:#2d3137;
    --l0:#24282e;--l1:#1e3c56;--l2:#2f6d9f;--l3:#4d9ed6;--l4:#8ccbf4;
    --warn:#f0a054;--warnbg:#2a2018;--flow:#4d9ed6;--sm:#2f6d9f;
  }}
  :root[data-theme="dark"]{
    --bg:#131519;--panel:#1b1e23;--ink:#e9e9e5;--dim:#9a9da4;--line:#2d3137;
    --l0:#24282e;--l1:#1e3c56;--l2:#2f6d9f;--l3:#4d9ed6;--l4:#8ccbf4;
    --warn:#f0a054;--warnbg:#2a2018;--flow:#4d9ed6;--sm:#2f6d9f;
  }
  text{font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;fill:var(--ink)}
  .h1{font-size:16px;font-weight:650}
  .lbl{font-size:12px;font-weight:650;fill:var(--dim);letter-spacing:.04em}
  .sub{font-size:11px;fill:var(--dim)}
  .mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px}
  .big{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:15px;font-weight:650}
  .c{rx:1.5;fill:var(--l0)}
  .box{fill:var(--panel);stroke:var(--line);stroke-width:1.25;rx:7}
  .warnbox{fill:var(--warnbg);stroke:var(--warn);stroke-width:1.25;rx:7}
  .arrow{stroke:var(--dim);stroke-width:1.5;fill:none;marker-end:url(#ar)}
  .warntx{fill:var(--warn);font-size:10.5px;font-weight:650}
  .fr{animation-duration:${LOOP}s;animation-timing-function:step-end;animation-iteration-count:infinite}
  ${FRAMES.map((_, i) => {
    const a = (i * 100 / FRAMES.length).toFixed(2), b = ((i + 1) * 100 / FRAMES.length).toFixed(2);
    return `@keyframes fr${i}{0%{opacity:${i === 0 ? 1 : 0}}${a}%{opacity:1}${b}%{opacity:0}100%{opacity:${i === 0 ? 1 : 0}}}\n  .fr${i}{animation-name:fr${i}}`;
  }).join('\n  ')}
  @keyframes tok{${tokenKF}}
  .tok{animation:tok ${CYCLE}s linear infinite}
  @keyframes pulse{0%,84%{opacity:.25}88%,96%{opacity:1}100%{opacity:.25}}
  .pulse{animation:pulse ${CYCLE}s linear infinite}
  @media (prefers-reduced-motion:reduce){
    .fr,.tok,.pulse,[class^="s"]{animation:none}
    .fr0{opacity:1}.fr1,.fr2,.fr3{opacity:0}
  }
  ${seqCSS}
</style>
<defs>
  <marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="var(--dim)"/></marker>
  <marker id="arf" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="var(--flow)"/></marker>
</defs>

<rect x="0" y="0" width="${W}" height="${H}" fill="var(--bg)"/>

<text class="h1" x="60" y="40">CUDA k-means · execution model</text>
${FRAMES.map((f, i) => frameText(i, 'mono', W - 60, 40, 'end',
  `<tspan fill="var(--dim)">iteration </tspan><tspan class="big">${f}</tspan><tspan fill="var(--dim)"> / ${r.trace.length}</tspan><tspan fill="var(--dim)">   δ </tspan><tspan class="big">${frames[i].delta.toFixed(5)}</tspan>`)).join('\n')}
<line x1="60" y1="54" x2="${W - 60}" y2="54" stroke="var(--line)"/>

<text class="lbl" x="60" y="84">HOST → DEVICE · ONCE, BEFORE THE LOOP</text>
<rect class="box" x="60" y="96" width="204" height="58"/>
<text class="mono" x="162" y="121" text-anchor="middle">objects[${fmt(r.N)}][${r.D}]</text>
<text class="sub" x="162" y="139" text-anchor="middle">point-major (host)</text>
<path class="arrow" d="M276,125 L360,125"/>
<text class="sub" x="318" y="115" text-anchor="middle" font-weight="650">transpose</text>
<text class="sub" x="318" y="143" text-anchor="middle">for coalescing</text>
<rect class="box" x="372" y="96" width="204" height="58"/>
<text class="mono" x="474" y="121" text-anchor="middle">objects[${r.D}][${fmt(r.N)}]</text>
<text class="sub" x="474" y="139" text-anchor="middle">coordinate-major (device)</text>
<path class="arrow" d="M588,125 L672,125"/>
<text class="sub" x="630" y="115" text-anchor="middle" font-weight="650">upload</text>
<rect class="box" x="684" y="96" width="156" height="58"/>
<text class="mono" x="762" y="121" text-anchor="middle">${(r.N * r.D * 4 / 1048576).toFixed(1)} MiB</text>
<text class="sub" x="762" y="139" text-anchor="middle">global memory</text>

<text class="lbl" x="60" y="184">GRID · ${fmt(r.numBlocks)} BLOCKS × ${r.blockSize} THREADS · ONE THREAD PER POINT</text>
<text class="sub" x="${W - 60}" y="184" text-anchor="end">shade = points in that block that changed cluster</text>
${cells.join('\n')}
${FRAMES.map((f, i) => frameText(i, 'sub', W - 60, GRID_Y + ROWS * PITCH + 18, 'end',
  `${fmt(frames[i].changed)} of ${fmt(r.N)} points changed`)).join('\n')}

<text class="lbl" x="60" y="386">SHARED MEMORY PER BLOCK · ${fmt(r.sharedBytes)} BYTES, PACKED BY HAND</text>
${smRegions.map(s => `<rect x="${s.x.toFixed(1)}" y="${SM_Y}" width="${Math.max(1, s.w - 1.5).toFixed(1)}" height="${SM_H}" rx="3" fill="var(--sm)" opacity="${s.bytes > 200 ? 0.9 : 0.55}"/>`).join('\n')}
${smRegions.map(s => (s.w > 56
  ? `<text class="mono" x="${(s.x + s.w / 2).toFixed(1)}" y="${SM_Y + 21}" text-anchor="middle" font-size="10.5" fill="var(--ink)">${esc(s.name)}</text><text class="sub" x="${(s.x + s.w / 2).toFixed(1)}" y="${SM_Y + SM_H + 15}" text-anchor="middle" font-size="10">${fmt(s.bytes)} B</text>`
  : `<text class="sub" x="${(s.x + s.w / 2).toFixed(1)}" y="${SM_Y + SM_H + 15}" text-anchor="middle" font-size="9.5">${fmt(s.bytes)}</text>`)).join('\n')}
<text class="sub" x="60" y="${SM_Y + SM_H + 32}">a single extern __shared__ allocation, sliced into six regions by pointer arithmetic</text>

<text class="lbl" x="60" y="478">THREE KERNELS PER ITERATION</text>
${kernels.map((k, i) => {
  const x = KB_X0 + i * (KB_W + KB_GAP);
  return `<rect class="${k.warn ? 'warnbox' : 'box'}" x="${x}" y="${KB_Y}" width="${KB_W}" height="${KB_H}"/>
<text class="mono" x="${x + KB_W / 2}" y="${KB_Y + 24}" text-anchor="middle" font-size="11.5">${esc(k.n)}</text>
<text class="sub" x="${x + KB_W / 2}" y="${KB_Y + 42}" text-anchor="middle">${esc(k.g)} threads</text>${
  k.warn ? `\n<text class="warntx" x="${x + KB_W / 2}" y="${KB_Y + 57}" text-anchor="middle">one block — most SMs idle</text>` : ''}${
  i < 2 ? `\n<path class="arrow" d="M${x + KB_W + 4},${KB_Y + KB_H / 2} L${x + KB_W + KB_GAP - 6},${KB_Y + KB_H / 2}"/>` : ''}`;
}).join('\n')}
<circle class="tok" cx="${KB_X0 + 20}" cy="${KB_Y + KB_H / 2}" r="5" fill="var(--flow)"/>
<path class="pulse" stroke="var(--flow)" stroke-width="2.5" fill="none" stroke-linecap="round" marker-end="url(#arf)"
  d="M${KB_X0 + 3 * KB_W + 2 * KB_GAP},${KB_Y + KB_H + 10} L${KB_X0 + 3 * KB_W + 2 * KB_GAP + 16},${KB_Y + KB_H + 10} L${KB_X0 + 3 * KB_W + 2 * KB_GAP + 16},${KB_Y + KB_H + 28} L${KB_X0 - 16},${KB_Y + KB_H + 28} L${KB_X0 - 16},${KB_Y + KB_H / 2 + 10}"/>
<text class="sub" x="${W / 2}" y="${KB_Y + KB_H + 46}" text-anchor="middle">blocking device→host copy of δ, centroid division on the host, then the next iteration</text>
</svg>
`;

fs.mkdirSync('assets', { recursive: true });
fs.writeFileSync('assets/pipeline.svg', svg);
console.log(`assets/pipeline.svg  ${(Buffer.byteLength(svg) / 1024).toFixed(1)} KB`);
console.log(`  ${r.numBlocks} blocks in a ${COLS}x${ROWS} grid`);
console.log(`  ${seqs.size} distinct activity sequences -> ${seqs.size} CSS rules (not ${r.numBlocks})`);
console.log(`  frames: iterations ${FRAMES.join(', ')} over a ${LOOP}s loop`);

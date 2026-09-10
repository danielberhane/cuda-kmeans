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

const W = 900, H = 678;

// Frame dwell, as weights. Iteration 1 is informationally uniform -- every
// block changes all its points -- so it reads instantly and needs less time.
// The later frames are sparse and take longer to scan, so they get more.
const FRAMES  = [1, 3, 8, 31];
const WEIGHTS = [15, 25, 30, 30];
const LOOP    = 16;
const CYCLE   = 4;                       // one pass of the kernel token

const r = runKMeans(readPoints('data/points_3d.txt'), 4);
const frames = FRAMES.map(i => r.trace[i - 1]);
const fmt = n => n.toLocaleString('en-US');
const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

// cumulative keyframe stops from the weights
const stops = [0];
for (let i = 0; i < WEIGHTS.length; i++) stops.push(stops[i] + WEIGHTS[i]);

// ---- grid --------------------------------------------------------------------
const COLS = 62, CELL = 10, GAP = 2, PITCH = CELL + GAP;
const ROWS = Math.ceil(r.numBlocks / COLS);
const GRID_W = COLS * PITCH - GAP, GRID_X = Math.round((W - GRID_W) / 2), GRID_Y = 200;

// Encoding. Per-block activity collapses ~40x over the run: every block changes
// all 128 of its points on iteration 1, but by iteration 31 the busiest block
// changes 3. A single absolute scale across that range pushes every late frame
// into one colour -- the first version did exactly that, rendering 619 and 85
// working blocks as apparently blank grids.
//
// So level 0 is absolute and means "this block did no work"; active blocks are
// scaled within their own frame. Convergence is then carried by how much of the
// grid turns grey, while the blocks still working stay legible. The caption says
// the shading is relative and each frame prints its absolute counts.
const activeRange = frames.map(f => {
  let lo = Infinity, hi = 0;
  for (const v of f.blockChanged) if (v > 0) { if (v < lo) lo = v; if (v > hi) hi = v; }
  return { lo: lo === Infinity ? 0 : lo, hi };
});
const level = (v, fi) => {
  if (v === 0) return 0;
  const { lo, hi } = activeRange[fi];
  if (hi === lo) return 4;
  return Math.max(1, Math.min(4, 1 + Math.floor((v - lo) / (hi - lo) * 4)));
};

const seqs = new Map();
const cells = [];
for (let b = 0; b < r.numBlocks; b++) {
  const seq = frames.map((f, fi) => level(f.blockChanged[b], fi)).join('');
  seqs.set(seq, (seqs.get(seq) ?? 0) + 1);
  cells.push(`<rect class="c s${seq}" x="${GRID_X + (b % COLS) * PITCH}" y="${GRID_Y + Math.floor(b / COLS) * PITCH}" width="${CELL}" height="${CELL}"/>`);
}
const seqCSS = [...seqs.keys()].map(seq => {
  const kf = seq.split('').map((lv, i) => `${stops[i]}%{fill:var(--l${lv})}`).join('');
  return `@keyframes k${seq}{${kf}}.s${seq}{animation:k${seq} ${LOOP}s step-end infinite}`;
}).join('\n  ');

// ---- shared memory -----------------------------------------------------------
// Four of the six regions are under 20px wide at this scale, so in-canvas labels
// collide (48/16/48 rendered as "481648"). The bar carries only the one label it
// can fit; the rest go in a legend beneath it.
const SM_X = 60, SM_W = W - 120, SM_Y = 398, SM_H = 32;
let acc = 0;
const smRegions = r.shared.map((s, i) => {
  const x = SM_X + (acc / r.sharedBytes) * SM_W, w = (s.bytes / r.sharedBytes) * SM_W;
  acc += s.bytes;
  return { ...s, x, w, shade: [4, 2, 1, 2, 1, 2][i] };
});
// Laid out on a fixed 3-column grid over two rows: a single row of six
// overflows the 780px content width.
const LEG_Y = SM_Y + SM_H + 22, LEG_COL = (W - 120) / 3, LEG_ROW = 17;
const legend = smRegions.map((s, i) => {
  const x = SM_X + (i % 3) * LEG_COL, y = LEG_Y + Math.floor(i / 3) * LEG_ROW;
  return `<rect x="${x}" y="${y - 8}" width="9" height="9" rx="2" fill="var(--l${s.shade})"/>` +
         `<text class="mono" x="${x + 15}" y="${y}" font-size="10.5" fill="var(--ink)">${esc(s.name)}</text>` +
         `<text class="sub" x="${x + 118}" y="${y}" font-size="10">${fmt(s.bytes)} B</text>`;
}).join('\n');
const LEG_BOTTOM = LEG_Y + LEG_ROW;

// ---- kernel pipeline ---------------------------------------------------------
// Execution order as launched in cuda_kmeans.cu's convergence loop.
// reduce_cluster_changed runs SECOND, and the blocking device->host copy of
// delta falls between it and reduce_coord_clusters -- the host stall interrupts
// the pipeline rather than following it.
const kernels = [
  { n: 'find_nearest_cluster',   g: `${fmt(r.numBlocks)} × ${r.blockSize}` },
  { n: 'reduce_cluster_changed', g: `1 × ${fmt(r.reductionThreads)}`, warn: true },
  { n: 'reduce_coord_clusters',  g: `${fmt(r.numBlocks)} × ${r.blockSize}` },
];
const KB_Y = 546, KB_H = 64, KB_W = 236, KB_GAP = 30;
const KB_X0 = Math.round((W - (KB_W * 3 + KB_GAP * 2)) / 2);
const KB_MID = KB_Y + KB_H / 2;
const RET_Y = KB_Y + KB_H + 26;                  // the return path's horizontal run
const RET_R = KB_X0 + 3 * KB_W + 2 * KB_GAP + 22;
const RET_L = KB_X0 - 22;

const tokenStops = [
  [0, 20], [30, KB_W - 20], [38, KB_W + KB_GAP + 20], [56, 2 * KB_W + KB_GAP - 20],
  [66, 2 * (KB_W + KB_GAP) + 20], [88, 3 * KB_W + 2 * KB_GAP - 20], [100, 3 * KB_W + 2 * KB_GAP - 20],
];
const tokenKF = tokenStops.map(([p, x]) => `${p}%{transform:translateX(${x - 20}px)}`).join('');

const frText = (i, cls, x, y, anchor, body) =>
  `<text class="${cls} fr fr${i}" x="${x}" y="${y}" text-anchor="${anchor}">${body}</text>`;

const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-labelledby="ttl dsc">
<title id="ttl">CUDA k-means execution model</title>
<desc id="dsc">${fmt(r.N)} points in ${r.D} dimensions are transposed to a coordinate-major layout for coalesced access, tiled across ${fmt(r.numBlocks)} thread blocks of ${r.blockSize} threads, and processed by three kernels per iteration. The grid shades each block by how much work it did; it goes grey as the algorithm converges over ${r.trace.length} iterations. The second kernel launches a single block, leaving most of the device idle, and a blocking copy of the convergence value back to the host stalls the pipeline before the third kernel runs.</desc>
<style>
  :root{
    --bg:#fbfbfa;--panel:#ffffff;--ink:#16171a;--dim:#6a6d73;--line:#dcdcd6;
    --l0:#d6d6cd;--l1:#7fadd8;--l2:#5590c4;--l3:#34719f;--l4:#17548f;
    --warn:#c2410c;--warnbg:#fdf1e7;--flow:#17548f;
  }
  @media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
    --bg:#131519;--panel:#1b1e23;--ink:#e9e9e5;--dim:#9a9da4;--line:#2d3137;
    --l0:#2b3138;--l1:#2f5f83;--l2:#3f8bbe;--l3:#5eaee0;--l4:#93d0f6;
    --warn:#f0a054;--warnbg:#2a2018;--flow:#5eaee0;
  }}
  :root[data-theme="dark"]{
    --bg:#131519;--panel:#1b1e23;--ink:#e9e9e5;--dim:#9a9da4;--line:#2d3137;
    --l0:#2b3138;--l1:#2f5f83;--l2:#3f8bbe;--l3:#5eaee0;--l4:#93d0f6;
    --warn:#f0a054;--warnbg:#2a2018;--flow:#5eaee0;
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
  .ret{stroke:var(--flow);stroke-width:1.75;fill:none;marker-end:url(#arf);opacity:.85}
  .warntx{fill:var(--warn);font-size:10.5px;font-weight:650}
  .fr{animation-duration:${LOOP}s;animation-timing-function:step-end;animation-iteration-count:infinite}
  ${FRAMES.map((_, i) => {
    const on = stops[i], off = stops[i + 1] ?? 100;
    const kf = i === 0
      ? `0%{opacity:1}${off}%{opacity:0}`
      : `0%{opacity:0}${on}%{opacity:1}` + (off < 100 ? `${off}%{opacity:0}` : '');
    return `@keyframes fr${i}{${kf}}\n  .fr${i}{animation-name:fr${i}}`;
  }).join('\n  ')}
  @keyframes tok{${tokenKF}}
  .tok{animation:tok ${CYCLE}s linear infinite}
  @keyframes pulse{0%,66%{opacity:.2}72%,92%{opacity:.9}100%{opacity:.2}}
  .pulse{animation:pulse ${CYCLE}s linear infinite}
  @media (prefers-reduced-motion:reduce){
    .fr,.tok,.pulse,[class^="s"]{animation:none}
    .fr0{opacity:1}.fr1,.fr2,.fr3{opacity:0}.pulse{opacity:.85}
  }
  ${seqCSS}
</style>
<defs>
  <marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="var(--dim)"/></marker>
  <marker id="arf" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5.5" markerHeight="5.5" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="var(--flow)"/></marker>
</defs>

<rect x="0" y="0" width="${W}" height="${H}" fill="var(--bg)"/>

<text class="h1" x="60" y="40">CUDA k-means · execution model</text>
${FRAMES.map((f, i) => frText(i, 'mono', W - 60, 40, 'end',
  `<tspan fill="var(--dim)">iteration </tspan><tspan class="big">${f}</tspan><tspan fill="var(--dim)"> / ${r.trace.length}</tspan><tspan fill="var(--dim)" dx="16">δ </tspan><tspan class="big">${frames[i].delta.toFixed(5)}</tspan>`)).join('\n')}
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
<text class="sub" x="${W - 60}" y="184" text-anchor="end">grey = block did no work · blue = relative activity this iteration</text>
${cells.join('\n')}
${FRAMES.map((f, i) => frText(i, 'sub', W - 60, GRID_Y + ROWS * PITCH + 20, 'end',
  `${fmt(frames[i].changed)} of ${fmt(r.N)} points changed · ${fmt(frames[i].blockChanged.reduce((a, v) => a + (v ? 0 : 1), 0))} of ${fmt(r.numBlocks)} blocks idle`)).join('\n')}

<text class="lbl" x="60" y="384">SHARED MEMORY PER BLOCK · ${fmt(r.sharedBytes)} BYTES, PACKED BY HAND</text>
${smRegions.map(s => `<rect x="${s.x.toFixed(1)}" y="${SM_Y}" width="${Math.max(1, s.w - 1.5).toFixed(1)}" height="${SM_H}" rx="3" fill="var(--l${s.shade})"/>`).join('\n')}
<text class="mono" x="${(smRegions[0].x + smRegions[0].w / 2).toFixed(1)}" y="${SM_Y + SM_H / 2 + 4}" text-anchor="middle" font-size="11" fill="var(--panel)">s_objects &#160;1,536 B &#160;·&#160; 81% of the budget</text>
${legend}
<text class="sub" x="60" y="${LEG_BOTTOM + 22}">a single extern __shared__ allocation, sliced into six regions by pointer arithmetic</text>

<text class="lbl" x="60" y="${KB_Y - 22}">THREE KERNELS PER ITERATION · IN LAUNCH ORDER</text>
${kernels.map((k, i) => {
  const x = KB_X0 + i * (KB_W + KB_GAP);
  return `<rect class="${k.warn ? 'warnbox' : 'box'}" x="${x}" y="${KB_Y}" width="${KB_W}" height="${KB_H}"/>
<text class="mono" x="${x + KB_W / 2}" y="${KB_Y + 24}" text-anchor="middle" font-size="11.5">${esc(k.n)}</text>
<text class="sub" x="${x + KB_W / 2}" y="${KB_Y + 42}" text-anchor="middle">${esc(k.g)} threads</text>${
  k.warn ? `\n<text class="warntx" x="${x + KB_W / 2}" y="${KB_Y + 57}" text-anchor="middle">one block — most SMs idle</text>` : ''}${
  i < 2 ? `\n<path class="arrow" d="M${x + KB_W + 5},${KB_MID} L${x + KB_W + KB_GAP - 7},${KB_MID}"/>` : ''}`;
}).join('\n')}
<text class="warntx" x="${KB_X0 + 2 * KB_W + KB_GAP + KB_GAP / 2}" y="${KB_Y - 6}" text-anchor="middle" font-size="9.5">δ → host</text>
<text class="warntx" x="${KB_X0 + 2 * KB_W + KB_GAP + KB_GAP / 2}" y="${KB_Y + KB_H + 12}" text-anchor="middle" font-size="9.5">stall</text>
<circle class="tok" cx="${KB_X0 + 20}" cy="${KB_MID}" r="5" fill="var(--flow)"/>

<path class="ret pulse" d="M${KB_X0 + 3 * KB_W + 2 * KB_GAP + 5},${KB_MID} H${RET_R - 8} Q${RET_R},${KB_MID} ${RET_R},${KB_MID + 8} V${RET_Y - 8} Q${RET_R},${RET_Y} ${RET_R - 8},${RET_Y} H${RET_L + 8} Q${RET_L},${RET_Y} ${RET_L},${RET_Y - 8} V${KB_MID + 8} Q${RET_L},${KB_MID} ${RET_L + 8},${KB_MID} H${KB_X0 - 5}"/>
<text class="sub" x="${W / 2}" y="${RET_Y + 17}" text-anchor="middle">centroid sums copied back, divided on the host, next iteration</text>
</svg>
`;

fs.mkdirSync('assets', { recursive: true });
fs.writeFileSync('assets/pipeline.svg', svg);
console.log(`assets/pipeline.svg  ${(Buffer.byteLength(svg) / 1024).toFixed(1)} KB`);
console.log(`  ${r.numBlocks} blocks, ${COLS}x${ROWS} grid, ${seqs.size} distinct sequences`);
console.log(`  frame stops: ${stops.slice(0, 4).join('% ')}%  (weights ${WEIGHTS.join('/')})`);

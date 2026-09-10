// Interactive demo: runs the real algorithm and draws what the GPU actually
// does with it. Every panel is derived from the trace -- the grid shades by
// per-block change counts, the shared-memory bar is to real byte scale.

const $ = (s) => document.querySelector(s);
const fmt = (n) => n.toLocaleString('en-US');

const worker = new Worker('worker.js', { type: 'module' });

const state = {
  run: null, iter: 0, playing: false, lastTick: 0,
  params: { dataset: 'benchmark', K: 4, threshold: 0.001, blockSize: 128 },
  axes: [0, 1],          // which two of the D dimensions the scatter plots
};

// ---- theme-aware palette, re-read whenever the theme changes ----------------
let PAL = {};
function readPalette() {
  const cs = getComputedStyle(document.documentElement);
  const v = (n) => cs.getPropertyValue(n).trim();
  PAL = {
    ink: v('--ink'), dim: v('--dim'), line: v('--line'), panel: v('--panel'),
    idle: v('--l0'), ramp: [v('--l1'), v('--l2'), v('--l3'), v('--l4')],
    warn: v('--warn'),
    clusters: [v('--c1'), v('--c2'), v('--c3'), v('--c4'), v('--c5'),
               v('--c6'), v('--c7'), v('--c8')],
  };
}
readPalette();
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
  readPalette(); draw();
});

// ---- canvas helper: size for devicePixelRatio ------------------------------
function fit(canvas) {
  const dpr = Math.min(devicePixelRatio || 1, 2);
  const r = canvas.getBoundingClientRect();
  const w = Math.max(1, Math.round(r.width * dpr));
  const h = Math.max(1, Math.round(r.height * dpr));
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w: r.width, h: r.height };
}

// Bounds are per-axis-pair, so they are recomputed whenever the selection changes.
function computeBounds() {
  const { coords, N, D } = state.run;
  const [ax, ay] = state.axes;
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (let i = 0; i < N; i++) {
    const a = coords[i * D + ax], b = coords[i * D + ay];
    if (a < x0) x0 = a; if (a > x1) x1 = a;
    if (b < y0) y0 = b; if (b > y1) y1 = b;
  }
  const mx = (x1 - x0) * 0.02, my = (y1 - y0) * 0.02;
  state.bounds = { x0: x0 - mx, x1: x1 + mx, y0: y0 - my, y1: y1 + my };
}

// ---- points -----------------------------------------------------------------
// 99,968 individual arc() calls would be far too slow; write pixels directly.
function drawPoints() {
  const c = $('#points');
  const { ctx, w, h } = fit(c);
  ctx.clearRect(0, 0, w, h);
  if (!state.run) return;

  const { coords, N, D } = state.run;
  const [ax, ay] = state.axes;
  const t = state.run.trace[state.iter];
  const dpr = Math.min(devicePixelRatio || 1, 2);
  const img = ctx.createImageData(c.width, c.height);
  const px = new Uint32Array(img.data.buffer);

  const rgba = PAL.clusters.map((col) => {
    const m = col.match(/#(\w{2})(\w{2})(\w{2})/);
    const [r, g, b] = m ? [parseInt(m[1], 16), parseInt(m[2], 16), parseInt(m[3], 16)] : [128, 128, 128];
    return (255 << 24) | (b << 16) | (g << 8) | r;      // little-endian RGBA
  });

  // Scale to the data's actual bounds -- the bundled set spans 0-100 in
  // dimension 1 but only 1-80 in dimension 2, so a hardcoded 0-100 box would
  // leave a fifth of the canvas permanently empty.
  const B = state.bounds;
  const pad = 8 * dpr;
  const sx = (c.width - 2 * pad) / (B.x1 - B.x0);
  const sy = (c.height - 2 * pad) / (B.y1 - B.y0);
  for (let i = 0; i < N; i++) {
    const x = pad + (coords[i * D + ax] - B.x0) * sx;
    const y = c.height - pad - (coords[i * D + ay] - B.y0) * sy;
    const xi = x | 0, yi = y | 0;
    if (xi < 0 || yi < 0 || xi >= c.width || yi >= c.height) continue;
    px[yi * c.width + xi] = rgba[t.membership[i] % rgba.length];
  }
  ctx.putImageData(img, 0, 0);

  // centroids on top
  ctx.save();
  ctx.scale(1 / dpr, 1 / dpr);
  for (let k = 0; k < state.run.K; k++) {
    const x = pad + (t.centroids[k * D + ax] - B.x0) * sx;
    const y = c.height - pad - (t.centroids[k * D + ay] - B.y0) * sy;
    ctx.beginPath(); ctx.arc(x, y, 6 * dpr, 0, 7);
    ctx.fillStyle = PAL.panel; ctx.fill();
    ctx.lineWidth = 2.5 * dpr; ctx.strokeStyle = PAL.clusters[k % PAL.clusters.length]; ctx.stroke();
  }
  ctx.restore();
}

// ---- grid -------------------------------------------------------------------
function drawGrid() {
  const c = $('#grid');
  const { ctx, w, h } = fit(c);
  ctx.clearRect(0, 0, w, h);
  if (!state.run) return;

  const { numBlocks, blockSize } = state.run;
  const bc = state.run.trace[state.iter].blockChanged;

  // choose a column count that fills the box at a sensible cell size
  let cols = Math.ceil(Math.sqrt(numBlocks * (w / h)));
  let pitch = w / cols;
  let rows = Math.ceil(numBlocks / cols);
  while (rows * pitch > h && cols < numBlocks) { cols++; pitch = w / cols; rows = Math.ceil(numBlocks / cols); }
  const size = Math.max(1, pitch - Math.min(2, pitch * 0.16));

  for (let b = 0; b < numBlocks; b++) {
    const v = bc[b];
    ctx.fillStyle = v === 0 ? PAL.idle
      : PAL.ramp[Math.min(3, Math.floor(Math.sqrt(v / blockSize) * 4))];
    ctx.fillRect((b % cols) * pitch, Math.floor(b / cols) * pitch, size, size);
  }
}

// ---- shared memory ----------------------------------------------------------
function drawShared() {
  const c = $('#shared');
  const { ctx, w, h } = fit(c);
  ctx.clearRect(0, 0, w, h);
  if (!state.run) return;

  const { shared, sharedBytes } = state.run;
  const barH = 34;
  const shades = [PAL.ramp[3], PAL.ramp[2], PAL.ramp[1], PAL.ramp[2], PAL.ramp[1], PAL.ramp[2]];
  let x = 0;
  ctx.font = '10.5px ui-monospace, Menlo, monospace';
  ctx.textBaseline = 'middle';
  shared.forEach((s, i) => {
    const bw = (s.bytes / sharedBytes) * w;
    ctx.fillStyle = shades[i % shades.length];
    ctx.fillRect(x, 0, Math.max(1, bw - 1.5), barH);
    if (bw > 70) {                       // only s_objects is ever this wide
      ctx.fillStyle = PAL.panel; ctx.textAlign = 'center';
      ctx.fillText(`${s.name}  ${fmt(s.bytes)} B`, x + bw / 2, barH / 2);
    }
    x += bw;
  });
  // The remaining five regions are a few pixels wide; label them in HTML.
  $('#sm-legend').innerHTML = shared.map((s, i) =>
    `<span class="lg"><i style="background:${shades[i % shades.length]}"></i>` +
    `<span class="mono">${s.name}</span> ${fmt(s.bytes)}&#8202;B</span>`).join('');
}

// ---- readouts ---------------------------------------------------------------
function drawStats() {
  if (!state.run) return;
  const r = state.run, t = r.trace[state.iter];
  $('#s-iter').textContent = `${t.iter} / ${r.trace.length}`;
  $('#s-delta').textContent = t.delta.toFixed(5);
  $('#s-changed').textContent = fmt(t.changed);
  $('#s-grid').textContent = `${fmt(r.numBlocks)} × ${r.blockSize}`;
  $('#s-reduce').textContent = `1 × ${fmt(r.reductionThreads)}`;
  $('#s-shared').textContent = `${fmt(r.sharedBytes)} B`;
  $('#s-sizes').textContent = t.counts.map((n) => fmt(Math.round(n))).join(' · ');
  $('#scrub').value = String(state.iter);
  $('#converged').hidden = state.iter !== r.trace.length - 1;
  const [ax, ay] = state.axes;
  const proj = $('#proj');
  if (proj) proj.textContent = r.D > 2
    ? `Showing dimensions ${ax + 1} and ${ay + 1} of ${r.D}. Any 2D view hides the rest, so two clusters separated only along a hidden dimension will appear to sit on top of each other — try the other pairs.`
    : `Showing dimensions ${ax + 1} and ${ay + 1}.`;
}

function draw() { drawPoints(); drawGrid(); drawShared(); drawStats(); }

// ---- run --------------------------------------------------------------------
function requestRun() {
  $('#busy').hidden = false;
  $('#play').disabled = true;
  worker.postMessage({ type: 'run', payload: state.params });
}

worker.onmessage = (e) => {
  const { type, payload } = e.data;
  if (type === 'loaded') {
    $('#s-n').textContent = `${fmt(payload.N)} × ${payload.D}D`;
    buildAxisPicker(payload.D);
    requestRun();
  }
  if (type === 'result') {
    state.run = payload;
    computeBounds();
    state.iter = 0;
    $('#scrub').max = String(payload.trace.length - 1);
    $('#s-time').textContent = `${payload.ms.toFixed(0)} ms`;
    $('#busy').hidden = true;
    $('#play').disabled = false;
    state.playing = true;
    state.lastTick = performance.now();
    draw();
  }
};

// ---- playback ---------------------------------------------------------------
const HOLD = [900, 620, 460, 340, 260];   // early iterations carry the change
function tick(now) {
  if (state.playing && state.run) {
    const hold = HOLD[Math.min(state.iter, HOLD.length - 1)];
    if (now - state.lastTick > hold) {
      state.lastTick = now;
      state.iter = (state.iter + 1) % state.run.trace.length;
      draw();
    }
  }
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);

// Every 2D view of D-dimensional data hides D-2 axes, and two clusters separated
// only along a hidden axis will sit on top of each other. Offering the other pairs
// is the fix; the caption below the plot names the one being shown.
function buildAxisPicker(D) {
  const box = $('#axes');
  if (!box) return;                       // never let a missing control break the page
  if (D < 3) { box.closest('.axrow').hidden = true; return; }
  const pairs = [];
  for (let a = 0; a < D; a++) for (let b = a + 1; b < D; b++) pairs.push([a, b]);
  box.innerHTML = pairs.slice(0, 6).map(([a, b], i) =>
    `<button class="axbtn${i === 0 ? ' on' : ''}" data-a="${a}" data-b="${b}">${a + 1}×${b + 1}</button>`).join('');
  box.querySelectorAll('.axbtn').forEach(btn => btn.addEventListener('click', () => {
    box.querySelectorAll('.axbtn').forEach(x => x.classList.remove('on'));
    btn.classList.add('on');
    state.axes = [+btn.dataset.a, +btn.dataset.b];
    computeBounds();
    draw();
  }));
}

// ---- controls ---------------------------------------------------------------
function bind(id, key, parse = Number, after = requestRun) {
  const el = $(id);
  el.addEventListener('input', () => {
    state.params[key] = parse(el.value);
    const out = $(`${id}-out`);
    if (out) out.textContent = el.value;
    after();
  });
}
bind('#k', 'K');
bind('#block', 'blockSize');
$('#threshold').addEventListener('input', (e) => {
  state.params.threshold = Number(e.target.value);
  $('#threshold-out').textContent = Number(e.target.value).toFixed(4);
  requestRun();
});
document.querySelectorAll('input[name=dataset]').forEach((el) => {
  el.addEventListener('change', () => { state.params.dataset = el.value; requestRun(); });
});
$('#play').addEventListener('click', () => {
  state.playing = !state.playing;
  $('#play').textContent = state.playing ? 'Pause' : 'Play';
  state.lastTick = performance.now();
});
$('#step').addEventListener('click', () => {
  if (!state.run) return;
  state.playing = false; $('#play').textContent = 'Play';
  state.iter = (state.iter + 1) % state.run.trace.length;
  draw();
});
$('#scrub').addEventListener('input', (e) => {
  state.playing = false; $('#play').textContent = 'Play';
  state.iter = Number(e.target.value);
  draw();
});
addEventListener('resize', draw);

worker.postMessage({ type: 'load', payload: { url: 'data/points_3d.bin' } });

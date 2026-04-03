import { BaseSimulation } from '../baseSimulation.js';

// ── Shared tree helpers ───────────────────────────────────────────
function gini(data) {
  if (!data.length) return 0;
  const p1 = data.filter(pt => pt.label === 1).length / data.length;
  return 1 - p1 * p1 - (1 - p1) * (1 - p1);
}

function majority(data) {
  const c1 = data.filter(pt => pt.label === 1).length;
  return c1 * 2 >= data.length ? 1 : 0;
}

function buildClassTree(data, depth, maxDepth, minLeaf, rng) {
  if (!data.length) return { label: 0 };
  const maj = majority(data);
  if (depth >= maxDepth || data.length < minLeaf || gini(data) < 1e-6) return { label: maj };

  const feature = rng() > 0.5 ? 'x' : 'y'; // random feature at each node
  const vals = [...new Set(data.map(pt => pt[feature]))].sort((a, b) => a - b);
  let bestGain = 0, best = null;
  const pG = gini(data);

  for (let i = 0; i < vals.length - 1; i++) {
    const t = (vals[i] + vals[i + 1]) / 2;
    const L = data.filter(pt => pt[feature] <= t);
    const R = data.filter(pt => pt[feature] > t);
    if (!L.length || !R.length) continue;
    const wG = (L.length * gini(L) + R.length * gini(R)) / data.length;
    if (pG - wG > bestGain) { bestGain = pG - wG; best = { feature, t, L, R }; }
  }
  if (!best) return { label: maj };
  return {
    feature: best.feature, threshold: best.t,
    left:  buildClassTree(best.L, depth + 1, maxDepth, minLeaf, rng),
    right: buildClassTree(best.R, depth + 1, maxDepth, minLeaf, rng),
  };
}

function predictClassTree(node, x, y) {
  if (node.label !== undefined) return node.label;
  const v = node.feature === 'x' ? x : y;
  return v <= node.threshold
    ? predictClassTree(node.left, x, y)
    : predictClassTree(node.right, x, y);
}

function mseFn(data) {
  if (!data.length) return 0;
  const m = data.reduce((s, pt) => s + pt.y, 0) / data.length;
  return data.reduce((s, pt) => s + (pt.y - m) ** 2, 0) / data.length;
}

function buildRegTree(data, depth, maxDepth, minLeaf) {
  if (!data.length) return { value: 0 };
  const meanY = data.reduce((s, pt) => s + pt.y, 0) / data.length;
  if (depth >= maxDepth || data.length < minLeaf) return { value: meanY };

  const vals = [...new Set(data.map(pt => pt.x))].sort((a, b) => a - b);
  let bestGain = 0, best = null;
  const pM = mseFn(data);

  for (let i = 0; i < vals.length - 1; i++) {
    const t = (vals[i] + vals[i + 1]) / 2;
    const L = data.filter(pt => pt.x <= t);
    const R = data.filter(pt => pt.x > t);
    if (!L.length || !R.length) continue;
    const wM = (L.length * mseFn(L) + R.length * mseFn(R)) / data.length;
    if (pM - wM > bestGain) { bestGain = pM - wM; best = { t, L, R }; }
  }
  if (!best) return { value: meanY };
  return {
    threshold: best.t,
    left:  buildRegTree(best.L, depth + 1, maxDepth, minLeaf),
    right: buildRegTree(best.R, depth + 1, maxDepth, minLeaf),
  };
}

function predictRegTree(node, x) {
  if (node.value !== undefined) return node.value;
  return x <= node.threshold
    ? predictRegTree(node.left, x)
    : predictRegTree(node.right, x);
}

// ── Random Forest Classification ──────────────────────────────────
export class RandomForestClassificationSimulation extends BaseSimulation {
  setup() {
    this.history = [];
    this.epoch   = 0;
    this.forest  = [];
    this._grid   = null;
    const { nPoints, seed, noiseLevel, datasetType } = this.params;
    this.points = this.generateClassDataset(datasetType || 'moons', nPoints, seed, noiseLevel ?? 0.08);
    this._rng = this.seededRandom((seed || 42) + 1337);
  }

  predict(x, y) {
    if (!this.forest.length) return 0;
    const votes = this.forest.reduce((s, tree) => s + predictClassTree(tree, x, y), 0);
    return votes * 2 >= this.forest.length ? 1 : 0;
  }

  step() {
    const nTrees = this.params.nTrees || 20;
    if (this.epoch >= nTrees) return;
    this.epoch++;
    const n = this.points.length;
    const bootstrap = Array.from({ length: n }, () => this.points[Math.floor(this._rng() * n)]);
    this.forest.push(
      buildClassTree(bootstrap, 0, this.params.maxDepth || 4, this.params.minLeafSize || 3, this._rng)
    );
    this._grid = null; // invalidate boundary cache
    this.history.push({ epoch: this.epoch, ...this.computeMetrics() });
  }

  computeMetrics() {
    const labels = this.points.map(pt => pt.label);
    const preds  = this.points.map(pt => this.predict(pt.x, pt.y));
    return this.computeClassificationMetrics(labels, preds);
  }

  _buildGrid(G) {
    const grid = [];
    for (let gx = 0; gx < G; gx++) {
      for (let gy = 0; gy < G; gy++) {
        const x = (gx / (G - 1)) * 2 - 1, y = (gy / (G - 1)) * 2 - 1;
        grid.push({ gx, gy, pred: this.predict(x, y) });
      }
    }
    return grid;
  }

  _drawBoundary(W, H) {
    const G = 50;
    if (!this._grid) this._grid = this._buildGrid(G);
    this._grid.forEach(({ gx, gy, pred }) => {
      this.ctx.fillStyle = pred === 1 ? 'rgba(29,78,216,.12)' : 'rgba(220,38,38,.12)';
      this.ctx.fillRect(gx * (W / G), H - (gy + 1) * (H / G), W / G + 1, H / G + 1);
    });
  }

  render() {
    const { width: W, height: H } = this.canvas;
    this.ctx.clearRect(0, 0, W, H);
    this.ctx.fillStyle = '#fff'; this.ctx.fillRect(0, 0, W, H);
    if (this.forest.length) this._drawBoundary(W, H);

    this.points.forEach(({ x, y, label }) => {
      const px = ((x + 1) / 2) * W, py = H - ((y + 1) / 2) * H;
      this.ctx.beginPath(); this.ctx.arc(px, py, 4.5, 0, Math.PI * 2);
      this.ctx.fillStyle   = label === 1 ? '#1565c0' : '#c62828'; this.ctx.fill();
      this.ctx.strokeStyle = '#fff'; this.ctx.lineWidth = 1.2; this.ctx.stroke();
    });

    const nTrees = this.params.nTrees || 20;
    const m      = this.forest.length ? this.computeMetrics() : null;
    this.ctx.save();
    this.ctx.fillStyle = 'rgba(255,255,255,.93)';
    this.ctx.beginPath(); this.ctx.roundRect(8, 8, 260, 70, 6); this.ctx.fill();
    this.ctx.strokeStyle = '#e2e8f0'; this.ctx.lineWidth = 1; this.ctx.stroke();
    [`Trees: ${this.forest.length} / ${nTrees}  |  Depth: ${this.params.maxDepth || 4}`,
     m ? `Accuracy: ${(m.accuracy * 100).toFixed(1)}%  F1: ${(m.f1 * 100).toFixed(1)}%`
       : 'Press Run to build forest',
    ].forEach((line, i) => {
      this.ctx.font      = i === 0 ? 'bold 12px sans-serif' : '11px sans-serif';
      this.ctx.fillStyle = i === 0 ? '#1e293b' : '#374151';
      this.ctx.textAlign = 'left'; this.ctx.fillText(line, 18, 26 + i * 17);
    });
    this.ctx.restore();

    if (m) {
      const labels = this.points.map(pt => pt.label);
      const preds  = this.points.map(pt => this.predict(pt.x, pt.y));
      this.drawConfusionMatrix(this.ctx, labels, preds, 10, H - 142, 58);
    }
  }
}

// ── Random Forest Regression ──────────────────────────────────────
export class RandomForestRegressionSimulation extends BaseSimulation {
  setup() {
    this.history = [];
    this.epoch   = 0;
    this.forest  = [];
    const { nPoints, seed, noiseLevel, datasetType } = this.params;
    this.points = this.generateRegressionDataset(datasetType || 'sine', nPoints, seed, noiseLevel ?? 0.2);
    this._rng = this.seededRandom((seed || 42) + 1337);
  }

  predict(x) {
    if (!this.forest.length) return 0;
    return this.forest.reduce((s, tree) => s + predictRegTree(tree, x), 0) / this.forest.length;
  }

  step() {
    const nTrees = this.params.nTrees || 20;
    if (this.epoch >= nTrees) return;
    this.epoch++;
    const n = this.points.length;
    const bootstrap = Array.from({ length: n }, () => this.points[Math.floor(this._rng() * n)]);
    this.forest.push(buildRegTree(bootstrap, 0, this.params.maxDepth || 4, this.params.minLeafSize || 3));
    this.history.push({ epoch: this.epoch, ...this.computeMetrics() });
  }

  computeMetrics() {
    const trues = this.points.map(pt => pt.y);
    const preds = this.points.map(pt => this.predict(pt.x));
    return this.computeRegressionMetrics(trues, preds);
  }

  render() {
    const { width: W, height: H } = this.canvas;
    const PAD = 36;
    this.ctx.clearRect(0, 0, W, H);
    this.ctx.fillStyle = '#fff'; this.ctx.fillRect(0, 0, W, H);

    const toX = x => PAD + ((x + 1) / 2) * (W - 2 * PAD);
    const toY = y => H - PAD - ((y + 1.2) / 2.4) * (H - 2 * PAD);
    const clampY = y => Math.max(-1.2, Math.min(1.2, y));

    this.ctx.strokeStyle = '#e2e8f0'; this.ctx.lineWidth = 1;
    this.ctx.beginPath(); this.ctx.moveTo(PAD, toY(0)); this.ctx.lineTo(W - PAD, toY(0)); this.ctx.stroke();
    this.ctx.beginPath(); this.ctx.moveTo(toX(0), PAD); this.ctx.lineTo(toX(0), H - PAD); this.ctx.stroke();

    if (this.forest.length) {
      this.ctx.beginPath(); this.ctx.strokeStyle = '#1d4ed8'; this.ctx.lineWidth = 2;
      for (let i = 0; i <= 300; i++) {
        const x = -1 + (i / 300) * 2;
        const y = clampY(this.predict(x));
        i === 0 ? this.ctx.moveTo(toX(x), toY(y)) : this.ctx.lineTo(toX(x), toY(y));
      }
      this.ctx.stroke();
    }

    this.points.forEach(({ x, y }) => {
      this.ctx.beginPath(); this.ctx.arc(toX(x), toY(clampY(y)), 3.5, 0, Math.PI * 2);
      this.ctx.fillStyle = '#64748b'; this.ctx.fill();
      this.ctx.strokeStyle = '#fff'; this.ctx.lineWidth = 1; this.ctx.stroke();
    });

    const m = this.forest.length ? this.computeMetrics() : null;
    this.ctx.save();
    this.ctx.fillStyle = 'rgba(255,255,255,.93)';
    this.ctx.beginPath(); this.ctx.roundRect(8, 8, 280, 70, 6); this.ctx.fill();
    this.ctx.strokeStyle = '#e2e8f0'; this.ctx.lineWidth = 1; this.ctx.stroke();
    [`Trees: ${this.forest.length} / ${this.params.nTrees || 20}  |  Depth: ${this.params.maxDepth || 4}`,
     m ? `MAE: ${m.mae.toFixed(3)}  RMSE: ${m.rmse.toFixed(3)}` : 'Press Run to build forest',
    ].forEach((line, i) => {
      this.ctx.font      = i === 0 ? 'bold 12px sans-serif' : '11px sans-serif';
      this.ctx.fillStyle = i === 0 ? '#1e293b' : '#374151';
      this.ctx.textAlign = 'left'; this.ctx.fillText(line, 18, 26 + i * 17);
    });
    this.ctx.restore();
  }
}

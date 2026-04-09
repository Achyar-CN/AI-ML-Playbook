import { BaseSimulation } from '../baseSimulation.js';

// ── LightGBM histogram helpers ────────────────────────────────────
// Key difference from XGBoost: leaf-wise (best-first) tree growth
// using histogram-based approximate split finding.

const N_BINS = 32;

function buildHist(recs, feat) {
  const vals = recs.map(r => r.pt[feat]);
  let min = vals[0], max = vals[0];
  for (const v of vals) { if (v < min) min = v; if (v > max) max = v; }
  if (min === max) return null;
  const binW = (max - min) / N_BINS;
  const bins = Array.from({ length: N_BINS }, () => ({ G: 0, H: 0 }));
  for (const r of recs) {
    const b = Math.min(N_BINS - 1, Math.floor((r.pt[feat] - min) / binW));
    bins[b].G += r.g; bins[b].H += r.h;
  }
  return { bins, min, binW };
}

function bestHistSplit(recs, lambda, gamma) {
  const G = recs.reduce((s, r) => s + r.g, 0);
  const H = recs.reduce((s, r) => s + r.h, 0);
  let bestGain = gamma, bestSplit = null;

  for (const feat of ['x', 'y']) {
    const hist = buildHist(recs, feat);
    if (!hist) continue;
    const { bins, min, binW } = hist;
    let GL = 0, HL = 0;
    for (let i = 0; i < bins.length - 1; i++) {
      GL += bins[i].G; HL += bins[i].H;
      const GR = G - GL, HR = H - HL;
      if (HL < 1e-10 || HR < 1e-10) continue;
      const gain = 0.5 * (GL*GL/(HL+lambda) + GR*GR/(HR+lambda) - G*G/(H+lambda)) - gamma;
      if (gain > bestGain) {
        bestGain = gain;
        bestSplit = { feat, threshold: min + (i + 1) * binW };
      }
    }
  }
  return bestSplit ? { ...bestSplit, gain: bestGain } : null;
}

// Leaf-wise (best-first) tree building up to maxLeaves leaves.
function lgbmBuildTree(recs, maxLeaves, lambda, gamma) {
  // Represent the tree as a list of {feature,threshold,leftId,rightId} internal nodes
  // plus leaves identified by id. Root has id=0.
  const leaves = [{ id: 0, recs }];
  const nodes  = [];
  let nextId   = 1;

  for (let step = 0; step < maxLeaves - 1; step++) {
    let bestLeafIdx = -1, bestSplit = null, bestGain = -Infinity;
    leaves.forEach((leaf, idx) => {
      if (leaf.recs.length < 4) return;
      const split = bestHistSplit(leaf.recs, lambda, gamma);
      if (split && split.gain > bestGain) {
        bestGain = split.gain; bestLeafIdx = idx; bestSplit = split;
      }
    });
    if (bestLeafIdx < 0) break;

    const leaf = leaves[bestLeafIdx];
    const L = leaf.recs.filter(r => r.pt[bestSplit.feat] <= bestSplit.threshold);
    const R = leaf.recs.filter(r => r.pt[bestSplit.feat] >  bestSplit.threshold);
    if (!L.length || !R.length) break;

    nodes.push({ id: leaf.id, feature: bestSplit.feat, threshold: bestSplit.threshold,
                 leftId: nextId, rightId: nextId + 1 });
    leaves.splice(bestLeafIdx, 1,
      { id: nextId,     recs: L },
      { id: nextId + 1, recs: R });
    nextId += 2;
  }

  // Build leaf value map
  const leafMap = {};
  for (const lf of leaves) {
    const G = lf.recs.reduce((s, r) => s + r.g, 0);
    const H = lf.recs.reduce((s, r) => s + r.h, 0);
    leafMap[lf.id] = { value: -G / (H + lambda) };
  }

  // Recursively assemble tree object
  const nodeMap = {};
  for (const n of nodes) nodeMap[n.id] = n;

  function buildTree(id) {
    if (leafMap[id]) return leafMap[id];
    const n = nodeMap[id];
    if (!n) return { value: 0 };
    return { feature: n.feature, threshold: n.threshold,
             left: buildTree(n.leftId), right: buildTree(n.rightId) };
  }
  return buildTree(0);
}

function lgbmPredict(node, x, y) {
  if (node.value !== undefined) return node.value;
  const v = node.feature === 'x' ? x : y;
  return v <= node.threshold ? lgbmPredict(node.left, x, y) : lgbmPredict(node.right, x, y);
}

function sigmoid(x) { return 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, x)))); }

// ── LightGBM Classification ───────────────────────────────────────
export class LightGBMClassificationSimulation extends BaseSimulation {
  setup() {
    this.history = [];
    this.epoch   = 0;
    this.trees   = [];
    this._grid   = null;
    const { nPoints, seed, noiseLevel, datasetType } = this.params;
    this.points = this.generateClassDataset(datasetType || 'spiral', nPoints, seed, noiseLevel ?? 0.08);
    this._F = this.points.map(() => 0);
  }

  predict(x, y) {
    const logit = this.trees.reduce((s, t) => s + (this.params.learningRate || 0.1) * lgbmPredict(t, x, y), 0);
    return sigmoid(logit) >= 0.5 ? 1 : 0;
  }

  step() {
    if (this.epoch >= (this.params.nRounds || 30)) return;
    const lr       = this.params.learningRate || 0.1;
    const lambda   = this.params.lambda || 1;
    const gamma    = this.params.gamma || 0;
    const maxLeaves = this.params.maxLeaves || 8;

    const recs = this.points.map((pt, i) => {
      const p = sigmoid(this._F[i]);
      return { pt, g: p - pt.label, h: p * (1 - p) };
    });

    const tree = lgbmBuildTree(recs, maxLeaves, lambda, gamma);
    this.trees.push(tree);
    this.points.forEach((pt, i) => { this._F[i] += lr * lgbmPredict(tree, pt.x, pt.y); });
    this._grid = null;
    this.epoch++;
    this.history.push({ epoch: this.epoch, ...this.computeMetrics() });
  }

  computeMetrics() {
    const labels = this.points.map(pt => pt.label);
    const preds  = this.points.map(pt => this.predict(pt.x, pt.y));
    return this.computeClassificationMetrics(labels, preds);
  }

  _buildGrid(G) {
    const grid = [];
    for (let gx = 0; gx < G; gx++)
      for (let gy = 0; gy < G; gy++) {
        const x = (gx / (G - 1)) * 2 - 1, y = (gy / (G - 1)) * 2 - 1;
        grid.push({ gx, gy, pred: this.predict(x, y) });
      }
    return grid;
  }

  render() {
    const { width: W, height: H } = this.canvas;
    this.ctx.clearRect(0, 0, W, H);
    this.ctx.fillStyle = '#fff'; this.ctx.fillRect(0, 0, W, H);

    if (this.trees.length) {
      const G = 50;
      if (!this._grid) this._grid = this._buildGrid(G);
      this._grid.forEach(({ gx, gy, pred }) => {
        this.ctx.fillStyle = pred === 1 ? 'rgba(29,78,216,.12)' : 'rgba(220,38,38,.12)';
        this.ctx.fillRect(gx * (W / G), H - (gy + 1) * (H / G), W / G + 1, H / G + 1);
      });
    }

    this.points.forEach(({ x, y, label }) => {
      const px = ((x + 1) / 2) * W, py = H - ((y + 1) / 2) * H;
      this.ctx.beginPath(); this.ctx.arc(px, py, 4.5, 0, Math.PI * 2);
      this.ctx.fillStyle   = label === 1 ? '#1565c0' : '#c62828'; this.ctx.fill();
      this.ctx.strokeStyle = '#fff'; this.ctx.lineWidth = 1.2; this.ctx.stroke();
    });

    const nRounds = this.params.nRounds || 30;
    const m = this.trees.length ? this.computeMetrics() : null;
    this.ctx.save();
    this.ctx.fillStyle = 'rgba(255,255,255,.93)';
    this.ctx.beginPath(); this.ctx.roundRect(8, 8, 300, 70, 6); this.ctx.fill();
    this.ctx.strokeStyle = '#e2e8f0'; this.ctx.lineWidth = 1; this.ctx.stroke();
    [`Trees: ${this.trees.length} / ${nRounds}  |  LR: ${(this.params.learningRate||0.1).toFixed(2)}  Leaves: ${this.params.maxLeaves||8}`,
     m ? `Accuracy: ${(m.accuracy*100).toFixed(1)}%  F1: ${(m.f1*100).toFixed(1)}%` : 'Press Run to boost',
    ].forEach((line, i) => {
      this.ctx.font = i === 0 ? 'bold 12px sans-serif' : '11px sans-serif';
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

// ── LightGBM Regression ───────────────────────────────────────────
export class LightGBMRegressionSimulation extends BaseSimulation {
  setup() {
    this.history = [];
    this.epoch   = 0;
    this.trees   = [];
    this._3d     = this._is3DReg;
    const { nPoints, seed, noiseLevel, datasetType } = this.params;
    this.points = this.generateRegressionDataset(datasetType || 'sine', nPoints, seed, noiseLevel ?? 0.2);
    this._F0    = this.points.reduce((s, pt) => s + pt.y, 0) / Math.max(this.points.length, 1);
    this._F     = this.points.map(() => this._F0);
  }

  predict(x, z) {
    const lr = this.params.learningRate || 0.1;
    return this._F0 + this.trees.reduce((s, t) => s + lr * lgbmPredict(t, x, this._3d ? (z ?? 0) : 0), 0);
  }

  step() {
    if (this.epoch >= (this.params.nRounds || 30)) return;
    const lr       = this.params.learningRate || 0.1;
    const lambda   = this.params.lambda || 1;
    const gamma    = this.params.gamma || 0;
    const maxLeaves = this.params.maxLeaves || 8;

    // Regression: g = F - y, h = 1 (MSE gradient/hessian); use z as second feature when 3D
    const recs = this.points.map((pt, i) => ({
      pt: { x: pt.x, y: this._3d ? (pt.z ?? 0) : 0 },
      g: this._F[i] - pt.y, h: 1
    }));
    const tree = lgbmBuildTree(recs, maxLeaves, lambda, gamma);
    this.trees.push(tree);
    this.points.forEach((pt, i) => { this._F[i] += lr * lgbmPredict(tree, pt.x, this._3d ? (pt.z ?? 0) : 0); });
    this.epoch++;
    this.history.push({ epoch: this.epoch, ...this.computeMetrics() });
  }

  computeMetrics() {
    return this.computeRegressionMetrics(this.points.map(pt => pt.y), this.points.map(pt => this.predict(pt.x, pt.z)));
  }

  render() {
    const { width: W, height: H } = this.canvas;
    const PAD = 36;
    this.ctx.clearRect(0, 0, W, H);
    this.ctx.fillStyle = '#fff'; this.ctx.fillRect(0, 0, W, H);
    const toX = x => PAD + ((x + 1) / 2) * (W - 2 * PAD);
    const toY = y => H - PAD - ((y + 1.2) / 2.4) * (H - 2 * PAD);
    const clamp = y => Math.max(-1.2, Math.min(1.2, y));

    this.ctx.strokeStyle = '#e2e8f0'; this.ctx.lineWidth = 1;
    this.ctx.beginPath(); this.ctx.moveTo(PAD, toY(0)); this.ctx.lineTo(W - PAD, toY(0)); this.ctx.stroke();
    this.ctx.beginPath(); this.ctx.moveTo(toX(0), PAD); this.ctx.lineTo(toX(0), H - PAD); this.ctx.stroke();

    if (this.trees.length) {
      this.ctx.beginPath(); this.ctx.strokeStyle = '#1d4ed8'; this.ctx.lineWidth = 2;
      for (let i = 0; i <= 300; i++) {
        const x = -1 + (i / 300) * 2, y = clamp(this.predict(x));
        i === 0 ? this.ctx.moveTo(toX(x), toY(y)) : this.ctx.lineTo(toX(x), toY(y));
      }
      this.ctx.stroke();
    }

    this.points.forEach(({ x, y }) => {
      this.ctx.beginPath(); this.ctx.arc(toX(x), toY(clamp(y)), 3.5, 0, Math.PI * 2);
      this.ctx.fillStyle = '#64748b'; this.ctx.fill();
      this.ctx.strokeStyle = '#fff'; this.ctx.lineWidth = 1; this.ctx.stroke();
    });

    const m = this.trees.length ? this.computeMetrics() : null;
    this.ctx.save();
    this.ctx.fillStyle = 'rgba(255,255,255,.93)';
    this.ctx.beginPath(); this.ctx.roundRect(8, 8, 310, 70, 6); this.ctx.fill();
    this.ctx.strokeStyle = '#e2e8f0'; this.ctx.lineWidth = 1; this.ctx.stroke();
    [`Trees: ${this.trees.length} / ${this.params.nRounds||30}  |  LR: ${(this.params.learningRate||0.1).toFixed(2)}  Leaves: ${this.params.maxLeaves||8}`,
     m ? `MAE: ${m.mae.toFixed(3)}  RMSE: ${m.rmse.toFixed(3)}` : 'Press Run to boost',
    ].forEach((line, i) => {
      this.ctx.font = i === 0 ? 'bold 12px sans-serif' : '11px sans-serif';
      this.ctx.fillStyle = i === 0 ? '#1e293b' : '#374151';
      this.ctx.textAlign = 'left'; this.ctx.fillText(line, 18, 26 + i * 17);
    });
    this.ctx.restore();
  }
}

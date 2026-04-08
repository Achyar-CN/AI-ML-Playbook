import { BaseSimulation } from '../baseSimulation.js';
import { dataStore } from '../core/dataStore.js';

// ── Shared XGBoost tree helpers ───────────────────────────────────
// Each record: { pt, g, h }  (pt = data point, g = gradient, h = hessian)
function xgbBuildNode(recs, depth, maxDepth, minLeaf, lambda, gamma) {
  const G = recs.reduce((s, r) => s + r.g, 0);
  const H = recs.reduce((s, r) => s + r.h, 0);
  const leafValue = -G / (H + lambda);

  if (depth >= maxDepth || recs.length < minLeaf * 2) return { value: leafValue };

  let bestGain = gamma, bestSplit = null;

  for (const feat of ['x', 'y']) {
    const sorted = recs.slice().sort((a, b) => a.pt[feat] - b.pt[feat]);
    let GL = 0, HL = 0;
    for (let i = 0; i < sorted.length - 1; i++) {
      GL += sorted[i].g; HL += sorted[i].h;
      if (sorted[i].pt[feat] === sorted[i + 1].pt[feat]) continue;
      const GR = G - GL, HR = H - HL;
      const gain = 0.5 * (GL * GL / (HL + lambda) + GR * GR / (HR + lambda) - G * G / (H + lambda)) - gamma;
      if (gain > bestGain) {
        bestGain = gain;
        bestSplit = { feat, threshold: (sorted[i].pt[feat] + sorted[i + 1].pt[feat]) / 2 };
      }
    }
  }

  if (!bestSplit) return { value: leafValue };
  const L = recs.filter(r => r.pt[bestSplit.feat] <= bestSplit.threshold);
  const R = recs.filter(r => r.pt[bestSplit.feat] >  bestSplit.threshold);
  if (!L.length || !R.length) return { value: leafValue };

  return {
    feature: bestSplit.feat, threshold: bestSplit.threshold,
    left:  xgbBuildNode(L, depth + 1, maxDepth, minLeaf, lambda, gamma),
    right: xgbBuildNode(R, depth + 1, maxDepth, minLeaf, lambda, gamma),
  };
}

function xgbPredict(node, x, y) {
  if (node.value !== undefined) return node.value;
  const v = node.feature === 'x' ? x : y;
  return v <= node.threshold ? xgbPredict(node.left, x, y) : xgbPredict(node.right, x, y);
}

function sigmoid(x) { return 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, x)))); }

// ── XGBoost Classification ────────────────────────────────────────
export class XGBoostClassificationSimulation extends BaseSimulation {
  setup() {
    this.history = [];
    this.epoch   = 0;
    this.trees   = [];
    this._grid   = null;
    const { nPoints, seed, noiseLevel, datasetType } = this.params;
    this.points = this.generateClassDataset(datasetType || 'moons', nPoints, seed, noiseLevel ?? 0.08);
    // F[i] = log-odds, initialised to 0
    this._F = this.points.map(() => 0);
  }

  predict(x, y) {
    const logit = this.trees.reduce((s, t) => s + (this.params.learningRate || 0.3) * xgbPredict(t, x, y), 0);
    return sigmoid(logit) >= 0.5 ? 1 : 0;
  }

  _score(x, y) {
    return this.trees.reduce((s, t) => s + (this.params.learningRate || 0.3) * xgbPredict(t, x, y), 0);
  }

  step() {
    if (this.epoch >= (this.params.nRounds || 20)) return;
    const lr     = this.params.learningRate || 0.3;
    const lambda = this.params.lambda || 1;
    const gamma  = this.params.gamma || 0;

    const recs = this.points.map((pt, i) => {
      const p = sigmoid(this._F[i]);
      return { pt, g: p - pt.label, h: p * (1 - p) };
    });

    const tree = xgbBuildNode(recs, 0, this.params.maxDepth || 3, this.params.minLeafSize || 2, lambda, gamma);
    this.trees.push(tree);
    this.points.forEach((pt, i) => { this._F[i] += lr * xgbPredict(tree, pt.x, pt.y); });
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

    const nRounds = this.params.nRounds || 20;
    const m = this.trees.length ? this.computeMetrics() : null;
    this.ctx.save();
    this.ctx.fillStyle = 'rgba(255,255,255,.93)';
    this.ctx.beginPath(); this.ctx.roundRect(8, 8, 280, 70, 6); this.ctx.fill();
    this.ctx.strokeStyle = '#e2e8f0'; this.ctx.lineWidth = 1; this.ctx.stroke();
    [`Trees: ${this.trees.length} / ${nRounds}  |  LR: ${(this.params.learningRate||0.3).toFixed(2)}  λ: ${(this.params.lambda||1).toFixed(1)}`,
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

// ── XGBoost Regression ────────────────────────────────────────────

export class XGBoostRegressionSimulation extends BaseSimulation {
  setup() {
    this.history = [];
    this.epoch   = 0;
    this.trees   = [];
    this._is3D   = dataStore.is3D && dataStore.type === 'regression';
    const { nPoints, seed, noiseLevel, datasetType } = this.params;
    this.points = this.generateRegressionDataset(datasetType || 'sine', nPoints, seed, noiseLevel ?? 0.2);
    this._F0    = this.points.reduce((s, pt) => s + pt.y, 0) / Math.max(this.points.length, 1);
    this._F     = this.points.map(() => this._F0);
  }

  predict(x, z) {
    const lr = this.params.learningRate || 0.3;
    const y2 = this._is3D ? (z ?? 0) : 0;
    return this._F0 + this.trees.reduce((s, t) => s + lr * xgbPredict(t, x, y2), 0);
  }

  step() {
    if (this.epoch >= (this.params.nRounds || 20)) return;
    const lr     = this.params.learningRate || 0.3;
    const lambda = this.params.lambda || 1;
    const gamma  = this.params.gamma || 0;

    // For regression: g = F-y, h = 1
    // In 3D mode: pt.z → tree's 'y' feature so splits use both x and z
    const recs = this.points.map((pt, i) => ({
      pt: { x: pt.x, y: this._is3D ? (pt.z ?? 0) : 0 },
      g: this._F[i] - pt.y, h: 1,
    }));
    const tree = xgbBuildNode(recs, 0, this.params.maxDepth || 3, this.params.minLeafSize || 2, lambda, gamma);
    this.trees.push(tree);
    this.points.forEach((pt, i) => {
      this._F[i] += lr * xgbPredict(tree, pt.x, this._is3D ? (pt.z ?? 0) : 0);
    });
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
    this.ctx.beginPath(); this.ctx.roundRect(8, 8, 290, 70, 6); this.ctx.fill();
    this.ctx.strokeStyle = '#e2e8f0'; this.ctx.lineWidth = 1; this.ctx.stroke();
    [`Trees: ${this.trees.length} / ${this.params.nRounds||20}  |  LR: ${(this.params.learningRate||0.3).toFixed(2)}  λ: ${(this.params.lambda||1).toFixed(1)}`,
     m ? `MAE: ${m.mae.toFixed(3)}  RMSE: ${m.rmse.toFixed(3)}` : 'Press Run to boost',
    ].forEach((line, i) => {
      this.ctx.font = i === 0 ? 'bold 12px sans-serif' : '11px sans-serif';
      this.ctx.fillStyle = i === 0 ? '#1e293b' : '#374151';
      this.ctx.textAlign = 'left'; this.ctx.fillText(line, 18, 26 + i * 17);
    });
    this.ctx.restore();
  }
}

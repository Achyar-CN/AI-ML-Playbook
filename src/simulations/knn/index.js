import { BaseSimulation } from '../baseSimulation.js';

// ── Shared helpers ────────────────────────────────────────────────
function euclidean(ax, ay, bx, by) {
  return Math.sqrt((ax - bx) ** 2 + (ay - by) ** 2);
}
function manhattan(ax, ay, bx, by) {
  return Math.abs(ax - bx) + Math.abs(ay - by);
}

// ── KNN Classification ────────────────────────────────────────────
export class KNNClassificationSimulation extends BaseSimulation {
  setup() {
    this.history = [];
    this.epoch   = 0;
    this._grid   = null; // cached decision boundary grid
    const { nPoints, seed, noiseLevel, datasetType } = this.params;
    this.points = this.generateClassDataset(datasetType || 'moons', nPoints, seed, noiseLevel ?? 0.08);
  }

  _distFn() {
    return this.params.distanceMetric === 'manhattan' ? manhattan : euclidean;
  }

  predict(x, y) {
    const k    = Math.min(this.params.k || 5, this.points.length);
    const dist = this._distFn();
    const sorted = this.points
      .map(pt => ({ d: dist(x, y, pt.x, pt.y), label: pt.label }))
      .sort((a, b) => a.d - b.d);
    const votes = sorted.slice(0, k).reduce((s, n) => s + n.label, 0);
    return votes * 2 >= k ? 1 : 0;
  }

  step() {
    if (this.epoch >= 1) return; // lazy learner — single pass
    this.epoch++;
    this._grid = null; // invalidate cache
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
        const x = (gx / (G - 1)) * 2 - 1;
        const y = (gy / (G - 1)) * 2 - 1;
        grid.push({ gx, gy, pred: this.predict(x, y) });
      }
    }
    return grid;
  }

  _drawBoundary(W, H) {
    const G = 40;
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

    if (this.epoch > 0) this._drawBoundary(W, H);

    this.points.forEach(({ x, y, label }) => {
      const px = ((x + 1) / 2) * W, py = H - ((y + 1) / 2) * H;
      this.ctx.beginPath(); this.ctx.arc(px, py, 4.5, 0, Math.PI * 2);
      this.ctx.fillStyle   = label === 1 ? '#1565c0' : '#c62828'; this.ctx.fill();
      this.ctx.strokeStyle = '#fff'; this.ctx.lineWidth = 1.2; this.ctx.stroke();
    });

    const k      = this.params.k || 5;
    const metric = this.params.distanceMetric || 'euclidean';
    const m      = this.epoch > 0 ? this.computeMetrics() : null;

    this.ctx.save();
    this.ctx.fillStyle = 'rgba(255,255,255,.93)';
    this.ctx.beginPath(); this.ctx.roundRect(8, 8, 240, 70, 6); this.ctx.fill();
    this.ctx.strokeStyle = '#e2e8f0'; this.ctx.lineWidth = 1; this.ctx.stroke();
    [`k = ${k}  |  ${metric}`,
     m ? `Accuracy: ${(m.accuracy * 100).toFixed(1)}%  F1: ${(m.f1 * 100).toFixed(1)}%`
       : 'Press Run to classify',
    ].forEach((line, i) => {
      this.ctx.font      = i === 0 ? 'bold 12px sans-serif' : '11px sans-serif';
      this.ctx.fillStyle = i === 0 ? '#1e293b' : '#374151';
      this.ctx.textAlign = 'left';
      this.ctx.fillText(line, 18, 26 + i * 17);
    });
    this.ctx.restore();

    if (m) {
      const labels = this.points.map(pt => pt.label);
      const preds  = this.points.map(pt => this.predict(pt.x, pt.y));
      this.drawConfusionMatrix(this.ctx, labels, preds, 10, H - 142, 58);
    }
  }
}

// ── KNN Regression ────────────────────────────────────────────────
export class KNNRegressionSimulation extends BaseSimulation {
  setup() {
    this.history = [];
    this.epoch   = 0;
    const { nPoints, seed, noiseLevel, datasetType } = this.params;
    this.points = this.generateRegressionDataset(datasetType || 'sine', nPoints, seed, noiseLevel ?? 0.2);
  }

  predict(x) {
    const k = Math.min(this.params.k || 5, this.points.length);
    const sorted = this.points
      .map(pt => ({ d: Math.abs(pt.x - x), y: pt.y }))
      .sort((a, b) => a.d - b.d);
    return sorted.slice(0, k).reduce((s, n) => s + n.y, 0) / k;
  }

  step() {
    if (this.epoch >= 1) return;
    this.epoch++;
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

    // Axes
    this.ctx.strokeStyle = '#e2e8f0'; this.ctx.lineWidth = 1;
    this.ctx.beginPath(); this.ctx.moveTo(PAD, toY(0)); this.ctx.lineTo(W - PAD, toY(0)); this.ctx.stroke();
    this.ctx.beginPath(); this.ctx.moveTo(toX(0), PAD); this.ctx.lineTo(toX(0), H - PAD); this.ctx.stroke();

    if (this.epoch > 0) {
      this.ctx.beginPath(); this.ctx.strokeStyle = '#1d4ed8'; this.ctx.lineWidth = 2;
      for (let i = 0; i <= 200; i++) {
        const x = -1 + (i / 200) * 2;
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

    const m = this.epoch > 0 ? this.computeMetrics() : null;
    this.ctx.save();
    this.ctx.fillStyle = 'rgba(255,255,255,.93)';
    this.ctx.beginPath(); this.ctx.roundRect(8, 8, 260, 70, 6); this.ctx.fill();
    this.ctx.strokeStyle = '#e2e8f0'; this.ctx.lineWidth = 1; this.ctx.stroke();
    [`k = ${this.params.k || 5}  |  ${this.params.distanceMetric || 'euclidean'}`,
     m ? `MAE: ${m.mae.toFixed(3)}  RMSE: ${m.rmse.toFixed(3)}` : 'Press Run to fit',
    ].forEach((line, i) => {
      this.ctx.font      = i === 0 ? 'bold 12px sans-serif' : '11px sans-serif';
      this.ctx.fillStyle = i === 0 ? '#1e293b' : '#374151';
      this.ctx.textAlign = 'left';
      this.ctx.fillText(line, 18, 26 + i * 17);
    });
    this.ctx.restore();
  }
}

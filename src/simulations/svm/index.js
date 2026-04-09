import { BaseSimulation } from '../baseSimulation.js';

// ── Feature maps ──────────────────────────────────────────────────
function phiClass(x, y, kernel) {
  return kernel === 'poly2' ? [1, x, y, x * x, y * y, x * y] : [1, x, y];
}
function phiReg(x, kernel) {
  return kernel === 'poly2' ? [1, x, x * x] : [1, x];
}
function dot(w, phi) { return phi.reduce((s, f, i) => s + w[i] * f, 0); }

// ── SVM Classification ────────────────────────────────────────────
export class SVMClassificationSimulation extends BaseSimulation {
  setup() {
    this.history = [];
    this.epoch   = 0;
    this._grid   = null;
    const { nPoints, seed, noiseLevel, datasetType } = this.params;
    this.points = this.generateClassDataset(datasetType || 'linear', nPoints, seed, noiseLevel ?? 0.08);
    this.w = new Array(this.params.kernel === 'poly2' ? 6 : 3).fill(0);
  }

  _score(x, y) { return dot(this.w, phiClass(x, y, this.params.kernel || 'linear')); }
  predict(x, y) { return this._score(x, y) >= 0 ? 1 : 0; }

  step() {
    const epochs = this.params.epochs || 200;
    if (this.epoch >= epochs) return;
    this.epoch++;
    const C  = this.params.C || 1.0;
    const lr = this.params.learningRate || 0.01;
    const n  = this.points.length;
    const kernel = this.params.kernel || 'linear';
    const grad = new Array(this.w.length).fill(0);

    // Weight decay (L2 on non-bias terms)
    for (let j = 1; j < this.w.length; j++) grad[j] += this.w[j] / n;

    this.points.forEach(({ x, y: vy, label }) => {
      const yi  = label === 1 ? 1 : -1;
      const phi = phiClass(x, vy, kernel);
      if (yi * dot(this.w, phi) < 1) {
        for (let j = 0; j < phi.length; j++) grad[j] -= (C / n) * yi * phi[j];
      }
    });

    for (let j = 0; j < this.w.length; j++) this.w[j] -= lr * grad[j];
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
        const d = this._score(x, y);
        grid.push({ gx, gy, d });
      }
    }
    return grid;
  }

  _drawBoundary(W, H) {
    const G = 50;
    if (!this._grid) this._grid = this._buildGrid(G);
    this._grid.forEach(({ gx, gy, d }) => {
      const alpha = Math.min(0.22, Math.abs(d) * 0.1 + 0.06);
      this.ctx.fillStyle = d >= 0 ? `rgba(29,78,216,${alpha})` : `rgba(220,38,38,${alpha})`;
      this.ctx.fillRect(gx * (W / G), H - (gy + 1) * (H / G), W / G + 1, H / G + 1);
    });

    // Draw linear boundary + margin lines
    if ((this.params.kernel || 'linear') === 'linear' && Math.abs(this.w[2]) > 0.001) {
      const drawLine = (offset, style, dashed) => {
        this.ctx.strokeStyle = style;
        this.ctx.lineWidth   = dashed ? 1 : 2;
        dashed ? this.ctx.setLineDash([4, 4]) : this.ctx.setLineDash([]);
        this.ctx.beginPath();
        let first = true;
        for (let i = 0; i <= 120; i++) {
          const x = -1 + (i / 120) * 2;
          const y = -(this.w[0] + offset + this.w[1] * x) / this.w[2];
          if (y < -1.05 || y > 1.05) { first = true; continue; }
          const px = ((x + 1) / 2) * W, py = H - ((y + 1) / 2) * H;
          first ? (this.ctx.moveTo(px, py), first = false) : this.ctx.lineTo(px, py);
        }
        this.ctx.stroke();
        this.ctx.setLineDash([]);
      };
      drawLine(0,  'rgba(30,41,59,.8)',   false);
      drawLine(1,  'rgba(100,116,139,.4)', true);
      drawLine(-1, 'rgba(100,116,139,.4)', true);
    }
  }

  render() {
    const { width: W, height: H } = this.canvas;
    this.ctx.clearRect(0, 0, W, H);
    this.ctx.fillStyle = '#fff'; this.ctx.fillRect(0, 0, W, H);
    this._drawBoundary(W, H);

    this.points.forEach(({ x, y, label }) => {
      const px = ((x + 1) / 2) * W, py = H - ((y + 1) / 2) * H;
      const yi     = label === 1 ? 1 : -1;
      const margin = yi * this._score(x, y);
      this.ctx.beginPath(); this.ctx.arc(px, py, 4.5, 0, Math.PI * 2);
      this.ctx.fillStyle   = label === 1 ? '#1565c0' : '#c62828'; this.ctx.fill();
      this.ctx.strokeStyle = margin < 1 ? '#f59e0b' : '#fff';
      this.ctx.lineWidth   = margin < 1 ? 2 : 1.2; this.ctx.stroke();
    });

    const kernel = this.params.kernel || 'linear';
    const m      = this.epoch > 0 ? this.computeMetrics() : null;
    this.ctx.save();
    this.ctx.fillStyle = 'rgba(255,255,255,.93)';
    this.ctx.beginPath(); this.ctx.roundRect(8, 8, 270, 80, 6); this.ctx.fill();
    this.ctx.strokeStyle = '#e2e8f0'; this.ctx.lineWidth = 1; this.ctx.stroke();
    [`Kernel: ${kernel}  C: ${this.params.C || 1}`,
     `Epoch: ${this.epoch} / ${this.params.epochs || 200}`,
     m ? `Acc: ${(m.accuracy * 100).toFixed(1)}%  F1: ${(m.f1 * 100).toFixed(1)}%` : 'Press Run to train SVM',
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

// ── SVR (SVM Regression) ──────────────────────────────────────────
export class SVRSimulation extends BaseSimulation {
  setup() {
    this.history = [];
    this.epoch   = 0;
    this._3d     = this._is3DReg;
    const { nPoints, seed, noiseLevel, datasetType } = this.params;
    this.points = this.generateRegressionDataset(datasetType || 'sine', nPoints, seed, noiseLevel ?? 0.2);
    const kernel = this.params.kernel || 'linear';
    const nW = this._3d ? (kernel === 'poly2' ? 6 : 3) : (kernel === 'poly2' ? 3 : 2);
    this.w = new Array(nW).fill(0);
  }

  _phi(x, z) {
    const kernel = this.params.kernel || 'linear';
    const zv = this._3d ? (z ?? 0) : 0;
    if (this._3d) return kernel === 'poly2' ? [1, x, zv, x*x, zv*zv, x*zv] : [1, x, zv];
    return kernel === 'poly2' ? [1, x, x*x] : [1, x];
  }

  predict(x, z) { return dot(this.w, this._phi(x, z)); }

  step() {
    const epochs = this.params.epochs || 200;
    if (this.epoch >= epochs) return;
    this.epoch++;
    const C   = this.params.C || 1.0;
    const eps = this.params.epsilon || 0.1;
    const lr  = this.params.learningRate || 0.01;
    const n   = this.points.length;
    const grad = new Array(this.w.length).fill(0);

    for (let j = 1; j < this.w.length; j++) grad[j] += this.w[j] / n;

    this.points.forEach(({ x, y, z }) => {
      const phi  = this._phi(x, z);
      const err  = dot(this.w, phi) - y;
      if (err > eps) {
        for (let j = 0; j < phi.length; j++) grad[j] += (C / n) * phi[j];
      } else if (err < -eps) {
        for (let j = 0; j < phi.length; j++) grad[j] -= (C / n) * phi[j];
      }
    });

    for (let j = 0; j < this.w.length; j++) this.w[j] -= lr * grad[j];
    this.history.push({ epoch: this.epoch, ...this.computeMetrics() });
  }

  computeMetrics() {
    const trues = this.points.map(pt => pt.y);
    const preds = this.points.map(pt => this.predict(pt.x, pt.z));
    return this.computeRegressionMetrics(trues, preds);
  }

  render() {
    const { width: W, height: H } = this.canvas;
    const PAD = 36;
    this.ctx.clearRect(0, 0, W, H);
    this.ctx.fillStyle = '#fff'; this.ctx.fillRect(0, 0, W, H);

    const toX    = x => PAD + ((x + 1) / 2) * (W - 2 * PAD);
    const toY    = y => H - PAD - ((y + 1.2) / 2.4) * (H - 2 * PAD);
    const clampY = y => Math.max(-1.2, Math.min(1.2, y));
    const eps    = this.params.epsilon || 0.1;

    this.ctx.strokeStyle = '#e2e8f0'; this.ctx.lineWidth = 1;
    this.ctx.beginPath(); this.ctx.moveTo(PAD, toY(0)); this.ctx.lineTo(W - PAD, toY(0)); this.ctx.stroke();
    this.ctx.beginPath(); this.ctx.moveTo(toX(0), PAD); this.ctx.lineTo(toX(0), H - PAD); this.ctx.stroke();

    if (this.epoch > 0) {
      const xs = Array.from({ length: 201 }, (_, i) => -1 + (i / 200) * 2);
      const ys = xs.map(x => this.predict(x));

      // Epsilon tube fill
      this.ctx.beginPath();
      xs.forEach((x, i) => {
        const py = toY(clampY(ys[i] + eps));
        i === 0 ? this.ctx.moveTo(toX(x), py) : this.ctx.lineTo(toX(x), py);
      });
      xs.slice().reverse().forEach((x, i) => {
        const ri  = xs.length - 1 - i;
        this.ctx.lineTo(toX(x), toY(clampY(ys[ri] - eps)));
      });
      this.ctx.closePath();
      this.ctx.fillStyle = 'rgba(29,78,216,.08)'; this.ctx.fill();

      // Main curve
      this.ctx.beginPath(); this.ctx.strokeStyle = '#1d4ed8'; this.ctx.lineWidth = 2;
      xs.forEach((x, i) => {
        i === 0 ? this.ctx.moveTo(toX(x), toY(clampY(ys[i])))
                : this.ctx.lineTo(toX(x), toY(clampY(ys[i])));
      });
      this.ctx.stroke();
    }

    // Points (yellow ring = outside epsilon tube = support vector)
    this.points.forEach(({ x, y }) => {
      const isSV = this.epoch > 0 && Math.abs(this.predict(x) - y) > eps;
      this.ctx.beginPath(); this.ctx.arc(toX(x), toY(clampY(y)), 3.5, 0, Math.PI * 2);
      this.ctx.fillStyle   = '#64748b'; this.ctx.fill();
      this.ctx.strokeStyle = isSV ? '#f59e0b' : '#fff';
      this.ctx.lineWidth   = isSV ? 1.5 : 1; this.ctx.stroke();
    });

    const m = this.epoch > 0 ? this.computeMetrics() : null;
    this.ctx.save();
    this.ctx.fillStyle = 'rgba(255,255,255,.93)';
    this.ctx.beginPath(); this.ctx.roundRect(8, 8, 300, 80, 6); this.ctx.fill();
    this.ctx.strokeStyle = '#e2e8f0'; this.ctx.lineWidth = 1; this.ctx.stroke();
    [`Kernel: ${this.params.kernel || 'linear'}  C: ${this.params.C || 1}  \u03b5: ${eps}`,
     `Epoch: ${this.epoch} / ${this.params.epochs || 200}`,
     m ? `MAE: ${m.mae.toFixed(3)}  RMSE: ${m.rmse.toFixed(3)}` : 'Press Run to train SVR',
    ].forEach((line, i) => {
      this.ctx.font      = i === 0 ? 'bold 12px sans-serif' : '11px sans-serif';
      this.ctx.fillStyle = i === 0 ? '#1e293b' : '#374151';
      this.ctx.textAlign = 'left'; this.ctx.fillText(line, 18, 26 + i * 17);
    });
    this.ctx.restore();
  }
}

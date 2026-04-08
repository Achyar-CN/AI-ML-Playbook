import { BaseSimulation } from '../baseSimulation.js';
import { dataStore } from '../core/dataStore.js';

export class AdaBoostSimulation extends BaseSimulation {
  setup() {
    this.history = [];
    this.epoch   = 0;
    this.stumps  = [];
    const { nPoints, seed, noiseLevel, datasetType } = this.params;

    // Adaboost uses ±1 labels
    this.points = this.generateClassDataset(datasetType || 'linear', nPoints, seed, noiseLevel ?? 0.08)
      .map(pt => ({ ...pt, label: pt.label===1 ? 1 : -1, weight: 1/nPoints }));
  }

  _predictStump(pt, s) {
    const val = s.feature==='x' ? pt.x : pt.y;
    return s.polarity * (val <= s.threshold ? 1 : -1);
  }

  _fitWeakLearner() {
    let bestErr = Infinity, bestStump = null;
    for (const feature of ['x','y']) {
      const vals = [...new Set(this.points.map(pt => pt[feature]))].sort((a,b)=>a-b);
      for (let i = 0; i < vals.length-1; i++) {
        const threshold = (vals[i]+vals[i+1])/2;
        for (const polarity of [1,-1]) {
          let err = 0;
          this.points.forEach(pt => {
            const val = feature==='x' ? pt.x : pt.y;
            if (polarity*(val<=threshold?1:-1) !== pt.label) err += pt.weight;
          });
          if (err < bestErr) { bestErr = err; bestStump = { feature, threshold, polarity }; }
        }
      }
    }
    return { stump: bestStump, error: bestErr };
  }

  step() {
    if (this.epoch >= this.params.epochs) return;
    const shrinkage = this.params.learningRate ?? 1.0;

    const { stump, error } = this._fitWeakLearner();
    if (!stump || error >= 0.5 - 1e-10) { this.epoch++; return; }

    const alpha = shrinkage * 0.5 * Math.log((1-error)/(error+1e-10));
    stump.alpha = alpha;
    this.stumps.push(stump);

    let Z = 0;
    this.points.forEach(pt => {
      pt.weight *= Math.exp(-alpha * pt.label * this._predictStump(pt, stump));
      Z += pt.weight;
    });
    this.points.forEach(pt => { pt.weight /= Z; });

    this.epoch++;
    this.history.push({ epoch: this.epoch, ...this.computeMetrics() });
  }

  predict(x, y) {
    if (this.stumps.length === 0) return 1;
    const score = this.stumps.reduce((s, st) => {
      const val = st.feature==='x' ? x : y;
      return s + st.alpha * st.polarity * (val <= st.threshold ? 1 : -1);
    }, 0);
    return score >= 0 ? 1 : -1;
  }

  render() {
    const { width: W, height: H } = this.canvas;
    this.ctx.clearRect(0, 0, W, H);
    this.ctx.fillStyle = '#fff'; this.ctx.fillRect(0, 0, W, H);

    // Decision regions
    const G = 60;
    for (let gx = 0; gx < G; gx++) {
      for (let gy = 0; gy < G; gy++) {
        const x = (gx/(G-1))*2-1, y = (gy/(G-1))*2-1;
        this.ctx.fillStyle = this.predict(x,y)===1 ? 'rgba(29,78,216,.13)' : 'rgba(220,38,38,.13)';
        this.ctx.fillRect(gx*(W/G), H-(gy+1)*(H/G), W/G+1, H/G+1);
      }
    }

    // Stump boundary lines
    const maxAlpha = Math.max(...this.stumps.map(s=>s.alpha), 1e-6);
    this.stumps.forEach(s => {
      const a = Math.min(1, .15 + (s.alpha/maxAlpha)*.55);
      this.ctx.strokeStyle = `rgba(245,124,0,${a})`;
      this.ctx.lineWidth = 1.2; this.ctx.setLineDash([4,3]);
      this.ctx.beginPath();
      if (s.feature==='x') {
        const px = ((s.threshold+1)/2)*W;
        this.ctx.moveTo(px,0); this.ctx.lineTo(px,H);
      } else {
        const py = H-((s.threshold+1)/2)*H;
        this.ctx.moveTo(0,py); this.ctx.lineTo(W,py);
      }
      this.ctx.stroke(); this.ctx.setLineDash([]);
    });

    // Points (size ∝ weight)
    const maxW = Math.max(...this.points.map(p=>p.weight), 1e-10);
    this.points.forEach(({ x, y, label, weight }) => {
      const px = ((x+1)/2)*W, py = H-((y+1)/2)*H;
      const r  = 3.5 + (weight/maxW)*6;
      this.ctx.beginPath(); this.ctx.arc(px, py, r, 0, Math.PI*2);
      this.ctx.fillStyle = label===1 ? '#1565c0' : '#c62828';
      this.ctx.globalAlpha = .85; this.ctx.fill(); this.ctx.globalAlpha = 1;
      this.ctx.strokeStyle = '#fff'; this.ctx.lineWidth = 1.2; this.ctx.stroke();
    });

    // Info panel
    const m = this.computeMetrics();
    const lastS = this.stumps[this.stumps.length-1];
    this.ctx.save();
    this.ctx.fillStyle = 'rgba(255,255,255,.93)';
    this.ctx.beginPath(); this.ctx.roundRect(8, 8, 240, 95, 6); this.ctx.fill();
    this.ctx.strokeStyle = '#e2e8f0'; this.ctx.lineWidth = 1; this.ctx.stroke();
    const lines = [
      `Round: ${this.epoch} / ${this.params.epochs}`,
      `Stumps: ${this.stumps.length}  |  Shrinkage: ${(this.params.learningRate??1).toFixed(2)}`,
      `Accuracy: ${(m.accuracy*100).toFixed(1)}%`,
      lastS ? `Last: ${lastS.feature}${lastS.polarity>0?'≤':'>'}${lastS.threshold.toFixed(2)}, α=${lastS.alpha.toFixed(3)}` : '',
    ];
    lines.forEach((line, i) => {
      if (!line) return;
      this.ctx.font      = i===0 ? 'bold 12px sans-serif' : '11px sans-serif';
      this.ctx.fillStyle = i===0 ? '#1e293b' : '#374151';
      this.ctx.textAlign = 'left';
      this.ctx.fillText(line, 18, 26+i*18);
    });
    this.ctx.restore();

    // Confusion matrix
    const labels = this.points.map(pt => pt.label===1?1:0);
    const preds  = this.points.map(pt => this.predict(pt.x,pt.y)===1?1:0);
    this.drawConfusionMatrix(this.ctx, labels, preds, 10, H-142, 58);
  }

  computeMetrics() {
    const labels = this.points.map(pt => pt.label===1?1:0);
    const preds  = this.points.map(pt => this.predict(pt.x,pt.y)===1?1:0);
    return this.computeClassificationMetrics(labels, preds);
  }
}

// ── Gradient Boosting Regression ──────────────────────────────────
// Fits decision stumps to residuals sequentially (gradient boosting with MSE loss).
export class GradientBoostingRegressionSimulation extends BaseSimulation {
  setup() {
    this.history = [];
    this.epoch   = 0;
    this.stumps  = [];
    this._is3D   = dataStore.is3D && dataStore.type === 'regression';
    const { nPoints, seed, noiseLevel, datasetType } = this.params;
    this.points  = this.generateRegressionDataset(datasetType || 'sine', nPoints, seed, noiseLevel ?? 0.2);
    this._F0     = this.points.reduce((s, pt) => s + pt.y, 0) / Math.max(this.points.length, 1);
    this._preds  = this.points.map(() => this._F0); // running predictions
  }

  _fitStump(residuals) {
    const features = this._is3D ? ['x', 'z'] : ['x'];
    let bestMse = Infinity, best = null;
    for (const feat of features) {
      const vals = [...new Set(this.points.map(pt => pt[feat] ?? 0))].sort((a, b) => a - b);
      for (let i = 0; i < vals.length - 1; i++) {
        const t = (vals[i] + vals[i + 1]) / 2;
        const L = [], R = [];
        this.points.forEach((pt, idx) => ((pt[feat] ?? 0) <= t ? L : R).push(residuals[idx]));
        if (!L.length || !R.length) continue;
        const mL = L.reduce((s, v) => s + v, 0) / L.length;
        const mR = R.reduce((s, v) => s + v, 0) / R.length;
        const mse = L.reduce((s, v) => s + (v - mL) ** 2, 0) + R.reduce((s, v) => s + (v - mR) ** 2, 0);
        if (mse < bestMse) { bestMse = mse; best = { feat, t, mL, mR }; }
      }
    }
    return best;
  }

  _stumpPredict(st, x, z) {
    const v = st.feat === 'z' ? (z ?? 0) : x;
    return v <= st.t ? st.mL : st.mR;
  }

  predict(x, z) {
    const lr = this.params.learningRate || 0.1;
    return this._F0 + this.stumps.reduce((s, st) => s + lr * this._stumpPredict(st, x, z), 0);
  }

  step() {
    if (this.epoch >= this.params.epochs) return;
    const lr = this.params.learningRate || 0.1;
    const residuals = this.points.map((pt, i) => pt.y - this._preds[i]);
    const stump = this._fitStump(residuals);
    if (stump) {
      this.stumps.push(stump);
      this.points.forEach((pt, i) => { this._preds[i] += lr * this._stumpPredict(stump, pt.x, pt.z); });
    }
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

    // Stump split lines (vertical, faint orange)
    this.ctx.setLineDash([4, 3]);
    this.ctx.strokeStyle = 'rgba(245,124,0,.4)'; this.ctx.lineWidth = 1;
    this.stumps.forEach(st => {
      const px = toX(st.t);
      this.ctx.beginPath(); this.ctx.moveTo(px, PAD); this.ctx.lineTo(px, H - PAD); this.ctx.stroke();
    });
    this.ctx.setLineDash([]);

    if (this.stumps.length) {
      this.ctx.beginPath(); this.ctx.strokeStyle = '#dc2626'; this.ctx.lineWidth = 2.5;
      for (let i = 0; i <= 400; i++) {
        const x = -1 + (i / 400) * 2, y = clamp(this.predict(x));
        i === 0 ? this.ctx.moveTo(toX(x), toY(y)) : this.ctx.lineTo(toX(x), toY(y));
      }
      this.ctx.stroke();
    }

    this.points.forEach(({ x, y }) => {
      this.ctx.beginPath(); this.ctx.arc(toX(x), toY(clamp(y)), 3.5, 0, Math.PI * 2);
      this.ctx.fillStyle = '#64748b'; this.ctx.fill();
      this.ctx.strokeStyle = '#fff'; this.ctx.lineWidth = 1; this.ctx.stroke();
    });

    const m = this.stumps.length ? this.computeMetrics() : null;
    this.ctx.save();
    this.ctx.fillStyle = 'rgba(255,255,255,.93)';
    this.ctx.beginPath(); this.ctx.roundRect(8, 8, 280, 70, 6); this.ctx.fill();
    this.ctx.strokeStyle = '#e2e8f0'; this.ctx.lineWidth = 1; this.ctx.stroke();
    [`Round: ${this.epoch} / ${this.params.epochs}  |  LR: ${(this.params.learningRate||0.1).toFixed(2)}`,
     m ? `MAE: ${m.mae.toFixed(3)}  RMSE: ${m.rmse.toFixed(3)}` : 'Press Run to boost',
    ].forEach((line, i) => {
      this.ctx.font = i === 0 ? 'bold 12px sans-serif' : '11px sans-serif';
      this.ctx.fillStyle = i === 0 ? '#1e293b' : '#374151';
      this.ctx.textAlign = 'left'; this.ctx.fillText(line, 18, 26 + i * 17);
    });
    this.ctx.restore();
  }
}
